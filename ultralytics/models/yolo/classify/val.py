# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score, auc
import wandb

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = []
        self.pred = []
        self.probs = []
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc, conf=self.args.conf, names=self.names.values(), task="classify"
        )
        self.pred = []
        self.targets = []
        self.probs = []

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())
        self.probs.append(preds.softmax(dim=1).detach().cpu())

    def finalize_metrics(self):
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def postprocess(self, preds):
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        results = self.metrics.results_dict

        y_true = torch.cat(self.targets).numpy()
        probs = torch.cat(self.probs).numpy()
        y_scores = probs[:, self.fake_class_index()]  # assumes binary classification

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1s = 2 * precision * recall / (precision + recall + 1e-16)
        best_f1_idx = np.argmax(f1s)
        best_threshold = thresholds[best_f1_idx]

        for t in [0.25, 0.5, 0.75, best_threshold]:
            preds_t = (y_scores >= t).astype(int)
            wandb.log({
                f"F1@{t:.2f}": f1_score(y_true, preds_t),
                f"Precision@{t:.2f}": precision_score(y_true, preds_t),
                f"Recall@{t:.2f}": recall_score(y_true, preds_t)
            })

        wandb.log({
            "PR AUC": auc(recall, precision),
            "ROC AUC": auc(*roc_curve(y_true, y_scores)[:2]),
            "PR Curve": wandb.plot.pr_curve(y_true, y_scores, labels=["fake"]),
            "ROC Curve": wandb.plot.roc_curve(y_true, y_scores, labels=["fake"])
        })

        return results

    def build_dataset(self, img_path):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def fake_class_index(self):
        return [k.lower() for k in self.names.values()].index("fake")
