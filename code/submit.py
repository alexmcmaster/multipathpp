import argparse

# change this if you have problem
import sys
#sys.path.insert(1, "~/.local/lib/python3.6/site-packages")


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path
import matplotlib.pyplot as plt

from submission_proto import motion_submission_pb2
#from train import WaymoLoader, Model
from model.data import get_dataloader, dict_to_cuda, normalize
from model.multipathpp import MultiPathPP
from prerender.utils.utils import get_config
from prerender.utils.visualize import parse_one_scene, plot_scene_prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to CNN model"
    )
    parser.add_argument(
        "--time-limit", type=int, required=False, default=80, help="Number time steps"
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save predictions"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config file path"
    )
    parser.add_argument(
        "--model-name", type=str, required=False, help="Model name"
    )

    parser.add_argument("--account-name", required=False, default="alex.r.mcmaster@gmail.com")
    parser.add_argument("--authors", required=False, default="Alex McMaster, Lars Ullrich")
    parser.add_argument("--method-name", required=False, default="multipathpp")
    parser.add_argument("--visualize", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, required=False, default=128)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = get_config(args.config)
    config["val"]["data_config"]["dataloader_config"]["batch_size"] = args.batch_size

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.model_path.endswith("pth"):
        #model = Model(args.model_name)
        model = MultiPathPP(config["model"])
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        model = torch.jit.load(args.model_path)

    model.cuda()
    model.eval()

    #dataset = WaymoLoader(args.test_data, is_test=True)
    #loader = DataLoader(
    #    dataset, batch_size=args.batch_size, num_workers=min(args.batch_size, 16)
    #)
    loader = get_dataloader(config["val"]["data_config"])

    num_steps = 0
    RES = {}
    with torch.no_grad():
        for data in tqdm(loader):
            if config["train"]["normalize"]:
                data = normalize(data, config)
            dict_to_cuda(data)
            probas, coords, covariance_matrices, loss_coeff = model(data, num_steps)
            if config["train"]["normalize"]:
                coords = coords * 10 + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()
                assert torch.isfinite(coords).all()
            assert torch.isfinite(coords).all()
            assert torch.isfinite(probas).all()
            assert torch.isfinite(covariance_matrices).all()
            for p, conf, aid, sid, c, y in zip(
                    coords, probas, data["agent_id"], data["scenario_id"],
                    data["shift"], data["yaw"]):
                if sid not in RES:
                    RES[sid] = list()
                RES[sid].append({"aid":aid, "conf":conf, "pred":p, "yaw":-y, "center": c})
                if args.visualize:
                    atype = int(data["target/agent_type"][0])
                    filename = f"scid_{sid}__aid_{aid}__atype_{atype}.npz"
                    filepath = os.path.join(args.test_data, filename)
                    print(filepath)
                    scene_data = np.load(filepath)
                    plot_scene_prediction(scene_data, p.cpu().numpy(), conf.cpu().numpy())
                    plt.show()
            num_steps += 1
        #for x, center, yaw, agent_id, scenario_id, _, _ in tqdm(loader):
        #    x = x.cuda()
        #    confidences_logits, logits = model(x)
        #    confidences = torch.softmax(confidences_logits, dim=1)

        #    logits = logits.cpu().numpy()
        #    confidences = confidences.cpu().numpy()
        #    agent_id = agent_id.cpu().numpy()
        #    center = center.cpu().numpy()
        #    yaw = yaw.cpu().numpy()
        #    for p, conf, aid, sid, c, y in zip(
        #        logits, confidences, agent_id, scenario_id, center, yaw
        #    ):
        #        if sid not in RES:
        #            RES[sid] = []

        #        RES[sid].append(
        #            {"aid": aid, "conf": conf, "pred": p, "yaw": -y, "center": c}
        #        )

    motion_challenge_submission = motion_submission_pb2.MotionChallengeSubmission()
    motion_challenge_submission.account_name = args.account_name
    motion_challenge_submission.authors.extend(args.authors.split(","))
    motion_challenge_submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    )
    motion_challenge_submission.unique_method_name = args.method_name

    selector = np.arange(4, args.time_limit + 1, 5)
    for scenario_id, data in tqdm(RES.items()):
        scenario_predictions = motion_challenge_submission.scenario_predictions.add()
        scenario_predictions.scenario_id = scenario_id
        prediction_set = scenario_predictions.single_predictions

        for d in data:
            predictions = prediction_set.predictions.add()
            predictions.object_id = int(d["aid"])

            y = d["yaw"]
            rot_matrix = np.array([
                [np.cos(y), -np.sin(y)],
                [np.sin(y), np.cos(y)],
            ])

            for i in np.argsort(-d["conf"].cpu()):
                scored_trajectory = predictions.trajectories.add()
                scored_trajectory.confidence = d["conf"][i]

                trajectory = scored_trajectory.trajectory

                p = d["pred"].cpu()[i][selector] @ rot_matrix + d["center"]

                trajectory.center_x.extend(p[:, 0])
                trajectory.center_y.extend(p[:, 1])

    with open(args.save, "wb") as f:
        f.write(motion_challenge_submission.SerializeToString())


if __name__ == "__main__":
    main()
