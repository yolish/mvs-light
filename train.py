import argparse, os, sys, time, gc, datetime
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models.nets import MVSLight, MVSLightLoss
from utils import *


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--device', default='0', help='select device')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')

# main function
def train(model, criterion, optimizer, train_loader, test_loader, start_epoch, args):
    milestones = [len(train_loader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(train_loader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(train_loader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, criterion, optimizer, sample, args)
            lr_scheduler.step()
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
                print(
                   "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                       epoch_idx, args.epochs, batch_idx, len(train_loader),
                       optimizer.param_groups[0]["lr"], loss,
                       scalar_outputs['mvs_depth_loss'],
                       time.time() - start_time))
            del scalar_outputs, image_outputs
        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
				'epoch': epoch_idx,
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict()},
				"{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(test_loader):
                start_time = time.time()
                global_step = len(train_loader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, criterion, sample, args)

                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                    print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                                                                        epoch_idx, args.epochs,
                                                                        batch_idx,
                                                                        len(test_loader), loss,
                                                                        scalar_outputs["mvs_depth_loss"],
                                                                        time.time() - start_time))
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs


            save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
            print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, criterion, test_loader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(test_loader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, criterion, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(test_loader), loss,
																	time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(test_loader), avg_test_scalars.mean()))
        print("final", avg_test_scalars.mean())


def train_sample(model, criterion, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    torch.cuda.set_device(torch.device("cuda:{}".format(args.device)))
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_min"],sample_cuda["depth_max"])
    depth_est = outputs["refined_depth"]
    depth_loss_init, depth_loss_mvs, depth_loss_refined, total_depth_loss = criterion(outputs, depth_gt, mask)

    total_depth_loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": total_depth_loss.item(),
                      "init_depth_loss": depth_loss_init.item(),
					  "mvs_depth_loss": depth_loss_mvs.item(),
					  "refined_depth_loss": depth_loss_refined.item(),
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    image_outputs = {"depth_est": depth_est * mask.to(dtype=torch.float32),
                     "depth_est_nomask": depth_est,
                     "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": mask.to(dtype=torch.float32),
                     "errormap": (depth_est - depth_gt).abs() * mask.to(dtype=torch.float32),
                     }

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)
3

@make_nograd_func
def test_sample_depth(model, criterion, sample, args):
    model.eval()
    torch.cuda.set_device(torch.device("cuda:{}".format(args.device)))
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]


    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_min"], sample_cuda["depth_max"])
    depth_est = outputs["refined_depth"]

    depth_loss_init, depth_loss_mvs, depth_loss_refined, total_depth_loss = criterion(outputs, depth_gt, mask)

    scalar_outputs = {"loss": total_depth_loss.item(),
                      "init_depth_loss": depth_loss_init.item(),
					  "mvs_depth_loss": depth_loss_mvs.item(),
					  "refined_depth_loss": depth_loss_refined.item(),
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": mask,
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }


    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

def test_sample(model, sample, args):
    model.eval()
    torch.cuda.set_device(args.device)
    sample_cuda = tocuda(sample)
    model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

def profile():
    warmup_iter = 5
    iter_dataloader = iter(test_loader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader))
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    set_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    # create logger for mode "train" and "testall"
    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)
        print("creating new summary file")
        logger = SummaryWriter(args.logdir)
    print("argv:", sys.argv[1:])
    print_args(args)

    # model, optimizer
    model = MVSLight(return_intermidiate=True)
    model.to(device)
    criterion = MVSLightLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
								pin_memory=args.pin_m)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
							   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, criterion, optimizer, train_loader, test_loader, start_epoch, args)
    elif args.mode == "test":
        test(model, criterion, test_loader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError
