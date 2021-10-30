import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='/data1/Datasets/dtu_eval/ucsnet_outputs/')
    parser.add_argument('--mode', type=str, default='pcd', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='/data1/Datasets/dtu_eval/SampleSet/MVS Data/')
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--vis_out_dir', type=str, default='eval_results_ucsnet')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()

    thresh = args.downsample_density
    pred_pcds = [os.path.join(args.outputs_dir, f) for f in os.listdir(args.outputs_dir) if f.endswith(".ply")]
    accuracy = []
    completeness = []
    overall_score = []
    for pred in pred_pcds:
        scan_id = int(os.path.basename(pred).split("net")[1].split("_")[0])
        print("Start evaluating scan: {} (file: {})".format(scan_id, pred))
        if args.visualize:
            pbar = tqdm(total=8)
        else:
            pbar = tqdm(total=7)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(pred)
        data_pcd = np.asarray(data_pcd_o3d.points)

        pbar.update(1)
        pbar.set_description('random shuffle pcd index')
        shuffle_rng = np.random.default_rng()
        shuffle_rng.shuffle(data_pcd, axis=0)

        pbar.update(1)
        pbar.set_description('downsample pcd')
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]

        pbar.update(1)
        pbar.set_description('masking data pcd')

        obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{scan_id}_10.mat')
        ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
        BB = BB.astype(np.float32)

        patch = args.patch_size
        inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
        data_in = data_down[inbound]

        data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
        grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
        data_grid_in = data_grid[grid_inbound]
        in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
        data_in_obs = data_in[grid_inbound][in_obs]

        pbar.update(1)
        pbar.set_description('read STL pcd')
        stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{scan_id:03}_total.ply')
        stl = np.asarray(stl_pcd.points)

        pbar.update(1)
        pbar.set_description('compute data2stl')
        nn_engine.fit(stl)
        dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
        max_dist = args.max_dist
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        pbar.update(1)
        pbar.set_description('compute stl2data')
        ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{scan_id}.mat')['P']

        stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
        above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
        stl_above = stl[above]

        nn_engine.fit(data_in)
        dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        if args.visualize:
            if os.path.exists(args.vis_out_dir):
                os.mkdir(args.vis_out_dir)
            pbar.update(1)
            pbar.set_description('visualize error')
            vis_dist = args.visualize_threshold
            R = np.array([[1,0,0]], dtype=np.float64)
            G = np.array([[0,1,0]], dtype=np.float64)
            B = np.array([[0,0,1]], dtype=np.float64)
            W = np.array([[1,1,1]], dtype=np.float64)
            data_color = np.tile(B, (data_down.shape[0], 1))
            data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
            data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
            data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
            write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
            stl_color = np.tile(B, (stl.shape[0], 1))
            stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
            stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
            stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
            write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

        pbar.update(1)
        pbar.set_description('done')
        pbar.close()
        over_all = (mean_d2s + mean_s2d) / 2
        print("Completed evaluating {}: acc: {}, completeness: {}, overall: {}".format(pred, mean_d2s, mean_s2d, over_all))
        accuracy.append(mean_d2s)
        completeness.append(mean_s2d)
        overall_score.append(over_all)


    print("Mean accuracy: {}, Mean completeness: {}, Mean overall: {}".format(np.mean(accuracy), np.mean(completeness), np.mean(overall_score)))
