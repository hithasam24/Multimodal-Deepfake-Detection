import torch
import torch.nn as nn
from .saff import VisualEncoder, AudioEncoder, SAFF
from .cmgan import CMGANBlock


class SAFFCMGANModel(nn.Module):
    def __init__(self, feat_dim=192):
        super().__init__()
        self.visual_encoder = VisualEncoder(feat_dim=feat_dim)
        self.audio_encoder = AudioEncoder(feat_dim=feat_dim)
        self.saff = SAFF(feat_dim=feat_dim)
        self.cmgan = CMGANBlock(feat_dim=feat_dim, num_heads=2, num_layers=1, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 4 + 1, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim * 2, 1),
        )

    def forward(self, frames, mel):
        v_seq = self.visual_encoder(frames)
        a_seq = self.audio_encoder(mel)

        saff_out = self.saff(v_seq, a_seq)
        v = saff_out["v_aligned"]
        a = saff_out["a_aligned"]
        fused = saff_out["fused_seq"]
        sync_score = saff_out["sync_score"]

        cmgan_out = self.cmgan(v, a)
        graph_repr = cmgan_out["graph_repr"]

        v_pool = v.mean(dim=1)
        a_pool = a.mean(dim=1)
        f_pool = fused.mean(dim=1)

        final_feat = torch.cat([v_pool, a_pool, f_pool, graph_repr, sync_score], dim=1)
        logits = self.classifier(final_feat).squeeze(1)

        return {
            "logits": logits,
            "sync_matrix": saff_out["sync_matrix"],
        }
