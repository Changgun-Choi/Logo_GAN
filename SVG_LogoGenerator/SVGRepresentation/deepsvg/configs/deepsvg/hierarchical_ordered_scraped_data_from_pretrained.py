from .default_icons import *
import os

class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False



class Config(Config):
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        datafolder = "../../../SVG_Data/"   # Make sure to set this to where your SVG_Data folder is located

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.dataloader_module = "deepsvg.svg_dataset"
        self.data_dir = os.path.join(datafolder, "data_for_training_deepsvg_model/WorldVector_SVGLogo_preprocessed_filtered_combined/")
        self.meta_filepath = os.path.join(datafolder,  "data_for_training_deepsvg_model/WorldVector_SVGLogo_preprocessed_filtered_combined_meta/meta_91788.csv")

        self.pretrained_path = os.path.join(datafolder,"deepsvg_pretrained_model/hierarchical_ordered.pth.tar")
        
        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 64 * num_gpus


        # Dataloader
        self.loader_num_workers = 4 * num_gpus

        # Training
        self.num_epochs = 150
       
        self.val_every = 100
        self.ckpt_every = 1000
        
    def set_train_vars(self, train_vars, dataloader):
        train_vars.x_inputs_train = [dataloader.dataset.get(idx, [*self.model_args, "tensor_grouped"])
                                     for idx in random.sample(range(len(dataloader.dataset)), k=100)]


