from dataclasses import dataclass

from megatron.core import parallel_state as mpu


@dataclass
class ParallelStates:
    """
    A dataclass that encapsulates the various parallel processing state information.

    This class stores the dimensions and ranks for different parallelism strategies including:
    tensor parallelism, pipeline parallelism, virtual pipeline parallelism,
    data/context parallelism, and expert parallelism.

    Attributes:
        tp_size: Tensor model parallel size
        tp_rank: Tensor model parallel rank
        pp_size: Pipeline model parallel size
        pp_rank: Pipeline model parallel rank
        vpp_size: Virtual pipeline model parallel size
        cp_size: Context/data parallel size
        cp_rank: Context/data parallel rank
        ep_size: Expert model parallel size
        ep_rank: Expert model parallel rank
        etp_size: Expert tensor parallel size
        etp_rank: Expert tensor parallel rank
    """

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    vpp_size: int = 1
    cp_size: int = 1
    cp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    etp_size: int = 1
    etp_rank: int = 0

    @classmethod
    def get_default_parallel_states(cls):
        """
        Creates and returns a ParallelStates instance with values from Megatron's parallel state.

        Returns:
            ParallelStates: An instance with current parallel state values from Megatron
        """
        return cls(
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
            cp_size=mpu.get_data_parallel_world_size(),
            cp_rank=mpu.get_data_parallel_rank(),
            ep_size=mpu.get_expert_model_parallel_world_size(),
            ep_rank=mpu.get_expert_model_parallel_rank(),
            etp_size=mpu.get_expert_tensor_parallel_world_size(),
            etp_rank=mpu.get_expert_tensor_parallel_rank(),
        )

    @classmethod
    def get_parallel_state(cls):
        """
        A convenience method that returns the default parallel states.

        Returns:
            ParallelStates: An instance with current parallel state values
        """
        return cls.get_default_parallel_states()
