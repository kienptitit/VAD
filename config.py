import os


class CFG:
    def __init__(self):
        self.tau_plus = 0.3
        self.temperature = 0.07

        self.mlp_ratio = 4
        self.n_heads = 1
        self.alpha = 0.1
        self.num_normal_labels = 20
        self.beta = 0.2
        self.tau = 0.1
        self.continue_training = False

        self.Batch_size = 16
        self.total_epochs = 600
        self.mle_normal_epochs = 50
        self.mle_epochs = 100

        self.MIL_epochs = 75
        self.sub_epochs = 12

        self.MLP_learning_rate = 0.01

        self.drop_out_mil = 0.0
        self.lr = 0.001
        self.lr_cosine = True
        self.lr_decay_rate = 0.1
        self.lr_decay_epochs = [50, 75, 90]
        self.out_feature_mil = 256

        self.n_clusters = 3

        self.checkpoint = None
        self.device = 'cuda'
        self.snippets = 32

        self.lr_warm = True
        self.lr_warm_epochs = 2
        self.save_result = True
        self.lr_warm_epochs = True
        self.mode_loss = 2
        self.init_nf_setting()
        self.init_log_dir()
        self.init_data_path()
        self.init_nf_model_hyper()
        self.init_nf_training_hyper_param()

    def init_nf_setting(self):
        self.clamp_alpha = 1.9
        self.normal_sub_epoch = 6
        self.focal_weighting = False
        self.normalizer = 10.0
        self.pos_beta = 0.4
        self.margin_abnormal_negative = 0.2 / self.normalizer
        self.margin_abnormal_positive = 0.1 / self.normalizer
        self.bgspp_lambda = 1.0

    def init_nf_training_hyper_param(self):
        self.NF_learning_rate = 0.001

    def init_data_path(self):
        self.train_path_MGFN = '/media/kiennguyen/Data/2023/NaverProject/UCF-Crime-10-Crop/Train_data_croped'
        self.test_path_MGFN = '/media/kiennguyen/Data/2023/NaverProject/UCF-Crime-10-Crop/Test_Cleaned'
        self.ucf_gt = '/media/kiennguyen/Data/2023/NaverProject/UCF-Crime-10-Crop/gt-ucf.npy'
        self.nf_save_score = '/media/kiennguyen/New Volume/PTIT/Naver/VAD/Figure'

    def init_nf_model_hyper(self):
        self.in_features = 1024
        self.pos_embed_dim = 128
        self.coupling_layers = 8
        self.flow_arch = 'conditional_flow_model'

    def init_constrastive_hyper(self):
        self.total_constrastive_epochs = 100

    def init_log_dir(self):
        self.log_path = None
        self.record_train_saved = r"train.pickle"
        self.record_test_saved = r"test.pickle"
        self.model_saved_path = r"model.pt"
        self.model_mil_saved_path = r"model_mil.pt"
        self.log_dir = None
