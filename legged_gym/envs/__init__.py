
from .base.legged_robot import LeggedRobot




from .go1.go1 import Go1
from .go1.go1_config_baseline import Go1BaseCfg, Go1BaseCfgPPO
from .go1.mixed_terrains.go1_rough_config import Go1RoughCfg, Go1RoughCfgPPO
from .go1.mixed_terrains.go1_obs_config import Go1ObsCfg, Go1ObsCfgPPO


from .go1.go1 import Go1

from .aliengo.aliengo import Aliengo
from .aliengo.aliengo_config_baseline import AliengoBaseCfg, AliengoBaseCfgPPO
# from .go1.flat.aliengo_flat_config import AliengoFlatCfg, AliengoFlatCfgPPO
from .aliengo.mixed_terrains.aliengo_rough_config import AliengoRoughCfg, AliengoRoughCfgPPO
from .aliengo.mixed_terrains.aliengo_obs_config import AliengoObsCfg, AliengoObsCfgPPO
# from .go1.mixed_terrains.aliengo_lbc_config import AliengoLbcCfg, AliengoLbcCfgPPO
from .aliengo.mixed_terrains.aliengo_enconder_config import AliengoEncoderCfg, AliengoEncoderCfgPPO
# from .go1.mixed_terrains.aliengo_his_config import AliengoHisCfg, AliengoHisCfgPPO


from legged_gym.utils.task_registry import task_registry





task_registry.register( "go1", Go1, Go1BaseCfg(), Go1BaseCfgPPO() )
task_registry.register( "go1_rough", Go1, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "go1_obs", Go1, Go1ObsCfg(), Go1ObsCfgPPO() )



task_registry.register( "aliengo", Aliengo, AliengoBaseCfg(), AliengoBaseCfgPPO() )
task_registry.register( "aliengo_rough", Aliengo, AliengoRoughCfg(), AliengoRoughCfgPPO() )
task_registry.register( "aliengo_obs", Aliengo, AliengoObsCfg(), AliengoObsCfgPPO() )
task_registry.register( "aliengo_encoder", Aliengo, AliengoEncoderCfg(), AliengoEncoderCfgPPO())



