import argparse
from os import path

from accelerate import Accelerator
import  evaluate
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm

from src.datasets import FFPP
from src.models import Detector


def collate_fn(batch):
    # just forward the dataset
    # only work with batch_size=1
    return batch[0]


@torch.no_grad()
def main(args):
    accelerator = Accelerator()

    # prepare model and dataset
    model = Detector().to(accelerator.device).eval()
    test_dataset = FFPP(args.data_root, model.transform, ['REAL', args.test_type], [args.compression])
    test_dataloader = accelerator.prepare(DataLoader(test_dataset, collate_fn=collate_fn))

    with accelerator.main_process_first():
        state_dict = torch.load(f'weights/realforensics_allbut{args.test_type.lower()}.pth', map_location='cpu')
        weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
        model.encoder.load_state_dict(weights_backbone)
        weights_df_head = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
        model.head.load_state_dict(weights_df_head)
    
    if accelerator.is_local_main_process:
        accuracy_calc = evaluate.load("accuracy")
        roc_auc_calc = evaluate.load("roc_auc")

    model.eval()
    progress_bar = tqdm(test_dataloader, disable=not accelerator.is_local_main_process)
    for clip, label in progress_bar:
        pred_prob = 1 - model(clip).sigmoid().mean()
        label = torch.tensor([label], device=accelerator.device)

        pred_label = (pred_prob > 0.5) * 1

        # sync across process
        pred_probs, pred_labels, labels = accelerator.gather_for_metrics((
        pred_prob, pred_label, label))

        if accelerator.is_local_main_process:
            accuracy_calc.add_batch(references=labels, predictions=pred_labels)
            roc_auc_calc.add_batch(references=labels, prediction_scores=pred_probs)

    if accelerator.is_local_main_process:
        accuracy = accuracy_calc.compute()['accuracy']
        roc_auc = roc_auc_calc.compute()['roc_auc']
        print(f'accuracy: {accuracy}, roc_auc: {roc_auc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealForensics evaluation on FaceForensics `leave-one-out` benchmark")
    parser.add_argument(
        "test_type",
        type=str,
        help="Type of manipulation to test on, should be one of (DF | F2F | FS | NT)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default='c23',
        help="Compression of testing videos, should be one of (raw | c23 | c40)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='/home/ernestchu/scratch4/OriginalDatasets/FF/',
        help="Directory to FaceForensics dataset"
    )
    
    args = parser.parse_args()
    assert args.test_type in ['DF', 'F2F', 'FS', 'NT'], 'test_type should be one of (DF | F2F | FS | NT)'
    assert args.compression in ['raw', 'c23', 'c40'], 'compression should be one of (raw | c23 | c40)'

    main(args)

