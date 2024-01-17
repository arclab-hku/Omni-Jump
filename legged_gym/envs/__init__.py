
from .base.legged_robot import LeggedRobot




from .go1.go1 import Go1
from .go1.go1_config_baseline import Go1BaseCfg, Go1BaseCfgPPO

from .aliengo.aliengo import Aliengo
from .aliengo.aliengo_config_baseline import AliengoBaseCfg, AliengoBaseCfgPPO

from .arcdog.arcdog import Arcdog
from .arcdog.arcdog_config_baseline import ArcdogBaseCfg, ArcdogBaseCfgPPO


from legged_gym.utils.task_registry import task_registry





task_registry.register( "go1", Go1, Go1BaseCfg(), Go1BaseCfgPPO() )



task_registry.register( "aliengo", Aliengo, AliengoBaseCfg(), AliengoBaseCfgPPO() )


task_registry.register( "arcdog", Arcdog, ArcdogBaseCfg(), ArcdogBaseCfgPPO() )