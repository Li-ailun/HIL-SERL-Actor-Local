########################################actor:


✅ 单臂复位完成！ pose=[0.21789999306201935, -0.3262999951839447, -0.15000000596046448, 0.0, 0.0, 0.0, 1.0], gripper=80.0, timeout=3.0
💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！
last return: 1.0:   0%|                    | 3270/1000000 [07:24<23:33:42, 11.75it/s] [actor-network-recv] recv_count=39, since_prev=9.226, leafs=81, total_params=8001041, checksum=8.336746, abs_mean=0.170633, sample_std=0.345709, head=[0.00034, -0.000205, 0.001242, 0.000255, -0.001972, 0.000815], shapes=[(6,), (256, 6), (6,), (256, 6)]
last return: 1.0:   0%|                    | 3368/1000000 [07:31<21:57:53, 12.60it/s] [actor-network-recv] recv_count=40, since_prev=7.744, leafs=81, total_params=8001041, checksum=8.338454, abs_mean=0.170680, sample_std=0.345695, head=[0.000302, -0.000248, 0.001278, 0.00011, -0.002125, 0.000771], shapes=[(6,), (256, 6), (6,), (256, 6)]
last return: 1.0:   0%|                    | 3470/1000000 [07:39<20:14:37, 13.67it/s]🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True
🎯 [VR归一化] 已设置当前手柄位姿为零点： left_pos_anchor=[-0.0729999989271164, 1.6888999938964844, 0.5734999775886536], right_pos_anchor=[0.18320000171661377, 1.658400058746338, 0.6003000140190125]
✅ 模式切换返回: success=True, message=Switched to VR control mode.
last return: 1.0:   0%|                    | 3480/1000000 [07:40<26:05:03, 10.61it/s] [actor-network-recv] recv_count=41, since_prev=8.874, leafs=81, total_params=8001041, checksum=8.343003, abs_mean=0.170636, sample_std=0.345610, head=[0.000313, -5e-05, 0.001224, 0.000255, -0.002172, 0.000866], shapes=[(6,), (256, 6), (6,), (256, 6)]
🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False
✅ 模式切换返回: success=True, message=Switched to IK service control mode.
last return: 1.0:   0%|                    | 3556/1000000 [07:46<20:28:42, 13.52it/s]🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True
🎯 [VR归一化] 已设置当前手柄位姿为零点： left_pos_anchor=[-0.065700002014637, 1.6791000366210938, 0.5651000142097473], right_pos_anchor=[0.2215999960899353, 1.5570000410079956, 0.6115000247955322]
✅ 模式切换返回: success=True, message=Switched to VR control mode.
last return: 1.0:   0%|                    | 3580/1000000 [07:48<20:22:02, 13.59it/s] [actor-episode-end] episode=12, step=3580, return=0.0000, env_return=0.0000, length=400, duration=31.459, success=0.0, intervention_count=2, intervention_steps=34, recv_count=41, applied_count=35
last return: 0.0:   0%|                    | 3580/1000000 [07:48<20:22:02, 13.59it/s]🤖 [USB Task Single Arm] 正在准备复位...
💡 【请按 VR 手柄的 Mode 2 键】切到脚本控制模式（会发送 use_vr_mode=False）
🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False
⏳ 已进入脚本控制，额外等待 2.00s 后开始复位...
✅ 模式切换返回: success=True, message=Switched to IK service control mode.
 [actor-network-recv] recv_count=42, since_prev=9.587, leafs=81, total_params=8001041, checksum=8.342869, abs_mean=0.170663, sample_std=0.345545, head=[0.000192, -0.000167, 0.00147, 0.000337, -0.002338, 0.000829], shapes=[(6,), (256, 6), (6,), (256, 6)]
🤖 [USB Task Single Arm] 开始向底层发送复位坐标...
🤖 正在执行单臂平滑复位/移动 (共 45 步)...
✅ 单臂复位完成！ pose=[0.21469999849796295, -0.3285999894142151, -0.15000000596046448, 0.0, 0.0, 0.0, 1.0], gripper=80.0, timeout=3.0
💡 【请按 VR 手柄的 Mode 0 键】重新夺回机械臂控制权，开始你的表演！
last return: 0.0:   0%|                    | 3606/1000000 [07:57<24:27:02, 11.32it/s]last return: 0.0:   0%|                    | 3608/1000000 [07:57<23:06:50, 11.97it/s]








###
###
###
###






#################################################learner

learner:   0%|                                        | 1980/1000000 [07:24<47:39:24,  5.82it/s] [learner-step] step=1980, publish_count=58, replay_buffer=3419, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029687920585274696}
learner:   0%|                                        | 1981/1000000 [07:24<47:08:15,  5.88it/s] [learner-step] step=1981, publish_count=58, replay_buffer=3419, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02952510491013527}
learner:   0%|                                        | 1982/1000000 [07:24<45:41:47,  6.07it/s] [learner-step] step=1982, publish_count=58, replay_buffer=3428, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03012079931795597}
learner:   0%|                                        | 1983/1000000 [07:24<47:13:43,  5.87it/s] [learner-step] step=1983, publish_count=58, replay_buffer=3428, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029110150411725044}
learner:   0%|                                        | 1984/1000000 [07:24<45:59:41,  6.03it/s] [learner-step] step=1984, publish_count=58, replay_buffer=3428, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029494324699044228}
learner:   0%|                                        | 1985/1000000 [07:24<45:21:20,  6.11it/s] [learner-step] step=1985, publish_count=58, replay_buffer=3437, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029566356912255287}
learner:   0%|                                        | 1986/1000000 [07:25<47:12:18,  5.87it/s] [learner-step] step=1986, publish_count=58, replay_buffer=3437, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030768461525440216}
learner:   0%|                                        | 1987/1000000 [07:25<47:50:58,  5.79it/s] [learner-step] step=1987, publish_count=58, replay_buffer=3437, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029729891568422318}
learner:   0%|                                        | 1988/1000000 [07:25<46:54:27,  5.91it/s] [learner-step] step=1988, publish_count=58, replay_buffer=3437, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030422350391745567}
learner:   0%|                                        | 1989/1000000 [07:25<48:20:38,  5.73it/s] [learner-step] step=1989, publish_count=58, replay_buffer=3446, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028616152703762054}
learner:   0%|                                        | 1990/1000000 [07:25<51:10:21,  5.42it/s] [learner-step] step=1990, publish_count=58, replay_buffer=3446, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028285855427384377}
learner:   0%|                                        | 1991/1000000 [07:25<49:17:46,  5.62it/s] [learner-step] step=1991, publish_count=58, replay_buffer=3446, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028597746044397354}
learner:   0%|                                        | 1992/1000000 [07:26<48:00:58,  5.77it/s] [learner-step] step=1992, publish_count=58, replay_buffer=3446, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029773052781820297}
learner:   0%|                                        | 1993/1000000 [07:26<46:52:12,  5.91it/s] [learner-step] step=1993, publish_count=58, replay_buffer=3454, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028786657378077507}
learner:   0%|                                        | 1994/1000000 [07:26<48:49:55,  5.68it/s] [learner-step] step=1994, publish_count=58, replay_buffer=3454, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030426230281591415}
learner:   0%|                                        | 1995/1000000 [07:26<47:45:58,  5.80it/s] [learner-step] step=1995, publish_count=58, replay_buffer=3454, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.0278336014598608}
learner:   0%|                                        | 1996/1000000 [07:26<46:34:16,  5.95it/s] [learner-step] step=1996, publish_count=58, replay_buffer=3454, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02948799543082714}
learner:   0%|                                        | 1997/1000000 [07:26<46:06:12,  6.01it/s] [learner-step] step=1997, publish_count=58, replay_buffer=3463, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030354764312505722}
learner:   0%|                                        | 1998/1000000 [07:27<46:50:16,  5.92it/s] [learner-step] step=1998, publish_count=58, replay_buffer=3463, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029669374227523804}
learner:   0%|                                        | 1999/1000000 [07:27<45:36:23,  6.08it/s] [learner-step] step=1999, publish_count=58, replay_buffer=3463, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02953476645052433}
learner:   0%|                                        | 2000/1000000 [07:27<45:11:13,  6.13it/s] [learner-network-publish] reason=train_periodic_update, step=2000, cost=0.174s, leafs=81, total_params=8001041, checksum=8.343003, abs_mean=0.170636, sample_std=0.345610, head=[0.000313, -5e-05, 0.001224, 0.000255, -0.002172, 0.000866], shapes=[(6,), (256, 6), (6,), (256, 6)]
 [learner-step] step=2000, publish_count=59, replay_buffer=3473, demo_buffer=7131, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028477363288402557}
I0426 14:53:14.553446 140438783083712 checkpoints.py:571] Saving checkpoint at step: 2000
I0426 14:53:14.554034 140438783083712 checkpoints.py:659] Using Orbax as backend to save Flax checkpoints. For potential troubleshooting see: https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#orbax-as-backend-troubleshooting
I0426 14:53:14.554417 140438783083712 pytree_checkpoint_handler.py:592] save_device_host_concurrent_bytes=None
I0426 14:53:14.555570 140438783083712 base_pytree_checkpoint_handler.py:441] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb9bde1c100>, enable_pinned_host_transfer=True, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0426 14:53:14.556187 140438783083712 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.36
I0426 14:53:14.557597 140438783083712 event_tracking.py:70] [process=0] [sync] Started save checkpoint @ /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.
I0426 14:53:14.560788 140438783083712 atomicity.py:140] Creating tmp directory /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp
I0426 14:53:14.563145 140438783083712 checkpoint.py:188] Wrote Metadata={'item_handlers': None, 'metrics': {}, 'performance_metrics': {}, 'init_timestamp_nsecs': 1777186394562197252, 'commit_timestamp_nsecs': None, 'custom_metadata': {}}, json={"item_handlers": null, "metrics": {}, "performance_metrics": {}, "init_timestamp_nsecs": 1777186394562197252, "commit_timestamp_nsecs": null, "custom_metadata": {}} to /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp/_CHECKPOINT_METADATA
I0426 14:53:14.865300 140438783083712 signaling_client.py:373] Using ThreadSafeKeyValueSignalingClient
I0426 14:53:14.866687 140438783083712 base_pytree_checkpoint_handler.py:154] [process=0][thread=MainThread] Initiated "orbax.checkpoint._src.serialization.type_handlers.NumpyHandler".serialize. Time taken: 0.216835s
I0426 14:53:14.867771 140438783083712 base_pytree_checkpoint_handler.py:130] [process=0] /jax/orbax/write/blocking_gbytes_per_sec: 1006.751 MiB/s (total gbytes: 305.2 MiB) (time elapsed: 0.30316901206970215 s) (per-host)
I0426 14:53:14.868052 140438783083712 base_pytree_checkpoint_handler.py:768] [process=0][thread=MainThread] Initiated Pytree async_save. Time taken: 0.303467s (batch_requests_ready=0.084477s, total_serialization_initiated=0.218019s, others=0.000970s)
I0426 14:53:14.868386 140438783083712 future.py:372] [process=0][thread=MainThread][operation_id=1] _SignalingThread.join() waiting for signals ([]) blocking the main thread will slow down blocking save times. This is likely due to main thread calling result() on a CommitFuture.
I0426 14:53:16.030066 140270611986176 array_metadata_store.py:248] [process=0][thread=write_metadata_after_commits] No metadata found for process_index=0, checkpoint_dir=/home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp. If the checkpoint does not contain jax.Array then it is expected. If checkpoint contains jax.Array then it should lead to an error eventually; if no error is raised then it is a bug.
I0426 14:53:16.104681 140270611986176 base_pytree_checkpoint_handler.py:1282] [process=0][thread=write_metadata_after_commits] Commit + Array metadata written. Time taken: 1.225183s (commit=1.149370s, array_metadata_write=0.075814s)
I0426 14:53:16.108296 140438783083712 base_pytree_checkpoint_handler.py:130] [process=0] /jax/orbax/write/gbytes_per_sec: 197.721 MiB/s (total gbytes: 305.2 MiB) (time elapsed: 1.543665885925293 s) (per-host)
I0426 14:53:16.110208 140438783083712 checkpoint.py:228] Read Metadata={'item_handlers': None, 'metrics': {}, 'performance_metrics': {}, 'init_timestamp_nsecs': 1777186394562197252, 'commit_timestamp_nsecs': None, 'custom_metadata': {}} from /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp/_CHECKPOINT_METADATA
I0426 14:53:16.111053 140438783083712 checkpoint.py:247] Updated Metadata={'item_handlers': 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler', 'metrics': {}, 'performance_metrics': {}, 'init_timestamp_nsecs': 1777186394562197252, 'commit_timestamp_nsecs': None, 'custom_metadata': {}} to /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp/_CHECKPOINT_METADATA
I0426 14:53:16.111465 140438783083712 event_tracking.py:125] [process=0] [sync] Finished blocking save in 1.55 seconds. Continuing save @ /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.
I0426 14:53:16.114908 140438783083712 array_metadata_store.py:284] [process=0][thread=MainThread] No metadata found for any process_index, checkpoint_dir=/home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp. Time taken: 0.000620s. If the checkpoint does not contain jax.Array then it is expected. If checkpoint contains jax.Array then it should lead to an error eventually; if no error is raised then it is a bug.
I0426 14:53:16.156669 140438783083712 base_pytree_checkpoint_handler.py:1406] [process=0][thread=MainThread] Pytree save finalize (merge_ocdbt + ArrayMetadata validation) completed. Time taken: 0.044348s. use_zarr3=False, enable_post_merge_validation=True, directory=/home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp
I0426 14:53:16.158388 140438783083712 atomicity.py:538] Renaming /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000.orbax-checkpoint-tmp to /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000
I0426 14:53:16.161501 140438783083712 atomicity.py:847] [process=0][thread=MainThread] Finished saving checkpoint (finalized tmp dir) to `/home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000`.
I0426 14:53:16.162798 140438783083712 event_tracking.py:138] [process=0] [sync] Finished save in 1.61 seconds @ /home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single/checkpoint_2000
 [learner-checkpoint-save] step=2000, cost=1.966s, path=/home/hddData/User/lixiang/hilserl_workspace/HIL-SERL-Project/examples/rlpd_checkpoints_single, leafs=81, total_params=8001041, checksum=8.343003, abs_mean=0.170636, sample_std=0.345610, head=[0.000313, -5e-05, 0.001224, 0.000255, -0.002172, 0.000866], shapes=[(6,), (256, 6), (6,), (256, 6)]
learner:   0%|                                       | 2001/1000000 [07:29<220:58:35,  1.25it/s] [learner-step] step=2001, publish_count=59, replay_buffer=3490, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028922751545906067}
learner:   0%|                                       | 2002/1000000 [07:29<172:36:41,  1.61it/s] [learner-step] step=2002, publish_count=59, replay_buffer=3490, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030070949345827103}
learner:   0%|                                       | 2003/1000000 [07:30<134:06:18,  2.07it/s] [learner-step] step=2003, publish_count=59, replay_buffer=3490, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02845171093940735}
learner:   0%|                                       | 2004/1000000 [07:30<107:27:28,  2.58it/s] [learner-step] step=2004, publish_count=59, replay_buffer=3499, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02922239527106285}
learner:   0%|                                        | 2005/1000000 [07:30<90:14:01,  3.07it/s] [learner-step] step=2005, publish_count=59, replay_buffer=3499, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030089201405644417}
learner:   0%|                                        | 2006/1000000 [07:30<77:00:26,  3.60it/s] [learner-step] step=2006, publish_count=59, replay_buffer=3499, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029343537986278534}
learner:   0%|                                        | 2007/1000000 [07:30<67:37:02,  4.10it/s] [learner-step] step=2007, publish_count=59, replay_buffer=3499, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029859721660614014}
learner:   0%|                                        | 2008/1000000 [07:30<60:32:24,  4.58it/s] [learner-step] step=2008, publish_count=59, replay_buffer=3508, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030360762029886246}
learner:   0%|                                        | 2009/1000000 [07:31<56:33:23,  4.90it/s] [learner-step] step=2009, publish_count=59, replay_buffer=3508, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02947787567973137}
learner:   0%|                                        | 2010/1000000 [07:31<52:24:21,  5.29it/s] [learner-step] step=2010, publish_count=59, replay_buffer=3508, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028746571391820908}
learner:   0%|                                        | 2011/1000000 [07:31<48:46:48,  5.68it/s] [learner-step] step=2011, publish_count=59, replay_buffer=3508, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029043525457382202}
learner:   0%|                                        | 2012/1000000 [07:31<46:27:44,  5.97it/s] [learner-step] step=2012, publish_count=59, replay_buffer=3517, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028997501358389854}
learner:   0%|                                        | 2013/1000000 [07:31<47:17:37,  5.86it/s] [learner-step] step=2013, publish_count=59, replay_buffer=3517, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02759527415037155}
learner:   0%|                                        | 2014/1000000 [07:31<45:39:14,  6.07it/s] [learner-step] step=2014, publish_count=59, replay_buffer=3517, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02990715019404888}
learner:   0%|                                        | 2015/1000000 [07:32<44:32:57,  6.22it/s] [learner-step] step=2015, publish_count=59, replay_buffer=3517, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029264360666275024}
learner:   0%|                                        | 2016/1000000 [07:32<44:26:49,  6.24it/s] [learner-step] step=2016, publish_count=59, replay_buffer=3525, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028395438566803932}
learner:   0%|                                        | 2017/1000000 [07:32<45:33:07,  6.09it/s] [learner-step] step=2017, publish_count=59, replay_buffer=3525, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02988196536898613}
learner:   0%|                                        | 2018/1000000 [07:32<44:19:48,  6.25it/s] [learner-step] step=2018, publish_count=59, replay_buffer=3525, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02896692417562008}
learner:   0%|                                        | 2019/1000000 [07:32<43:15:50,  6.41it/s] [learner-step] step=2019, publish_count=59, replay_buffer=3525, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02911822311580181}
learner:   0%|                                        | 2020/1000000 [07:32<43:46:44,  6.33it/s] [learner-step] step=2020, publish_count=59, replay_buffer=3534, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027738168835639954}
learner:   0%|                                        | 2021/1000000 [07:33<47:10:43,  5.88it/s] [learner-step] step=2021, publish_count=59, replay_buffer=3534, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03072396107017994}
learner:   0%|                                        | 2022/1000000 [07:33<46:29:04,  5.96it/s] [learner-step] step=2022, publish_count=59, replay_buffer=3534, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02908848412334919}
learner:   0%|                                        | 2023/1000000 [07:33<46:07:17,  6.01it/s] [learner-step] step=2023, publish_count=59, replay_buffer=3534, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030575942248106003}
learner:   0%|                                        | 2024/1000000 [07:33<45:22:31,  6.11it/s] [learner-step] step=2024, publish_count=59, replay_buffer=3534, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027540167793631554}
learner:   0%|                                        | 2025/1000000 [07:33<45:32:57,  6.09it/s] [learner-step] step=2025, publish_count=59, replay_buffer=3543, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02983950264751911}
learner:   0%|                                        | 2026/1000000 [07:33<44:48:55,  6.19it/s] [learner-step] step=2026, publish_count=59, replay_buffer=3543, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02917785942554474}
learner:   0%|                                        | 2027/1000000 [07:34<43:58:10,  6.30it/s] [learner-step] step=2027, publish_count=59, replay_buffer=3543, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029126092791557312}
learner:   0%|                                        | 2028/1000000 [07:34<43:25:38,  6.38it/s] [learner-step] step=2028, publish_count=59, replay_buffer=3543, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030186619609594345}
learner:   0%|                                        | 2029/1000000 [07:34<43:10:14,  6.42it/s] [learner-step] step=2029, publish_count=59, replay_buffer=3543, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028974831104278564}
learner:   0%|                                        | 2030/1000000 [07:34<42:48:40,  6.48it/s] [learner-step] step=2030, publish_count=59, replay_buffer=3554, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030029375106096268}
learner:   0%|                                        | 2031/1000000 [07:34<44:45:27,  6.19it/s] [learner-step] step=2031, publish_count=59, replay_buffer=3554, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028476065024733543}
learner:   0%|                                        | 2032/1000000 [07:34<44:17:07,  6.26it/s] [learner-step] step=2032, publish_count=59, replay_buffer=3554, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029192615300416946}
learner:   0%|                                        | 2033/1000000 [07:34<43:50:12,  6.32it/s] [learner-step] step=2033, publish_count=59, replay_buffer=3554, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03005100227892399}
learner:   0%|                                        | 2034/1000000 [07:35<44:28:32,  6.23it/s] [learner-step] step=2034, publish_count=59, replay_buffer=3564, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03003212809562683}
learner:   0%|                                        | 2035/1000000 [07:35<44:01:22,  6.30it/s] [learner-step] step=2035, publish_count=59, replay_buffer=3564, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029773829504847527}
learner:   0%|                                        | 2036/1000000 [07:35<43:40:11,  6.35it/s] [learner-step] step=2036, publish_count=59, replay_buffer=3564, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030821634456515312}
learner:   0%|                                        | 2037/1000000 [07:35<43:11:46,  6.42it/s] [learner-step] step=2037, publish_count=59, replay_buffer=3564, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029408669099211693}
learner:   0%|                                        | 2038/1000000 [07:35<42:58:41,  6.45it/s] [learner-step] step=2038, publish_count=59, replay_buffer=3564, demo_buffer=7142, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02947353385388851}
learner:   0%|                                        | 2039/1000000 [07:35<42:53:35,  6.46it/s] [learner-step] step=2039, publish_count=59, replay_buffer=3573, demo_buffer=7148, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029281247407197952}
learner:   0%|                                        | 2040/1000000 [07:36<45:11:29,  6.13it/s] [learner-step] step=2040, publish_count=59, replay_buffer=3573, demo_buffer=7148, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028439439833164215}
learner:   0%|                                        | 2041/1000000 [07:36<44:36:12,  6.21it/s] [learner-step] step=2041, publish_count=59, replay_buffer=3573, demo_buffer=7148, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029892779886722565}
learner:   0%|                                        | 2042/1000000 [07:36<44:17:48,  6.26it/s] [learner-step] step=2042, publish_count=59, replay_buffer=3573, demo_buffer=7148, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02803393453359604}
learner:   0%|                                        | 2043/1000000 [07:36<44:09:43,  6.28it/s] [learner-step] step=2043, publish_count=59, replay_buffer=3573, demo_buffer=7148, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029293781146407127}
learner:   0%|                                        | 2044/1000000 [07:36<44:15:12,  6.26it/s] [learner-step] step=2044, publish_count=59, replay_buffer=3584, demo_buffer=7159, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029074624180793762}
learner:   0%|                                        | 2045/1000000 [07:36<47:45:12,  5.81it/s] [learner-step] step=2045, publish_count=59, replay_buffer=3584, demo_buffer=7159, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03051290661096573}
learner:   0%|                                        | 2046/1000000 [07:37<47:05:17,  5.89it/s] [learner-step] step=2046, publish_count=59, replay_buffer=3584, demo_buffer=7159, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02808874472975731}
learner:   0%|                                        | 2047/1000000 [07:37<46:49:08,  5.92it/s] [learner-env-stats] step=2047, return=0.0000, length=400, duration=31.459, success=0.0, intervention_count=2, intervention_steps=34
 [learner-step] step=2047, publish_count=59, replay_buffer=3584, demo_buffer=7159, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.031108906492590904}
learner:   0%|                                        | 2048/1000000 [07:37<48:51:30,  5.67it/s] [learner-step] step=2048, publish_count=59, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029781240969896317}
learner:   0%|                                        | 2049/1000000 [07:37<47:51:42,  5.79it/s] [learner-step] step=2049, publish_count=59, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02993159368634224}
learner:   0%|                                        | 2050/1000000 [07:37<46:42:13,  5.94it/s] [learner-network-publish] reason=train_periodic_update, step=2050, cost=0.137s, leafs=81, total_params=8001041, checksum=8.342869, abs_mean=0.170663, sample_std=0.345545, head=[0.000192, -0.000167, 0.00147, 0.000337, -0.002338, 0.000829], shapes=[(6,), (256, 6), (6,), (256, 6)]
 [learner-step] step=2050, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.03002733178436756}
learner:   0%|                                        | 2051/1000000 [07:38<55:27:29,  5.00it/s] [learner-step] step=2051, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028056640177965164}
learner:   0%|                                        | 2052/1000000 [07:38<52:51:39,  5.24it/s] [learner-step] step=2052, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02905905805528164}
learner:   0%|                                        | 2053/1000000 [07:38<50:45:03,  5.46it/s] [learner-step] step=2053, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029592089354991913}
learner:   0%|                                        | 2054/1000000 [07:38<49:39:03,  5.58it/s] [learner-step] step=2054, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029717260971665382}
learner:   0%|                                        | 2055/1000000 [07:38<47:54:31,  5.79it/s] [learner-step] step=2055, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02913566119968891}
learner:   0%|                                        | 2056/1000000 [07:38<47:10:35,  5.88it/s] [learner-step] step=2056, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030031384900212288}
learner:   0%|                                        | 2057/1000000 [07:39<45:59:30,  6.03it/s] [learner-step] step=2057, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02874426729977131}
learner:   0%|                                        | 2058/1000000 [07:39<45:20:24,  6.11it/s] [learner-step] step=2058, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02870055101811886}
learner:   0%|                                        | 2059/1000000 [07:39<45:25:32,  6.10it/s] [learner-step] step=2059, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02860819548368454}
learner:   0%|                                        | 2060/1000000 [07:39<45:40:48,  6.07it/s] [learner-step] step=2060, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029417144134640694}
learner:   0%|                                        | 2061/1000000 [07:39<45:09:27,  6.14it/s] [learner-step] step=2061, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.0297828558832407}
learner:   0%|                                        | 2062/1000000 [07:39<45:00:03,  6.16it/s] [learner-step] step=2062, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02992626465857029}
learner:   0%|                                        | 2063/1000000 [07:39<44:54:35,  6.17it/s] [learner-step] step=2063, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029247021302580833}
learner:   0%|                                        | 2064/1000000 [07:40<44:16:03,  6.26it/s] [learner-step] step=2064, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029731445014476776}
learner:   0%|                                        | 2065/1000000 [07:40<43:54:33,  6.31it/s] [learner-step] step=2065, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02842472866177559}
learner:   0%|                                        | 2066/1000000 [07:40<43:23:04,  6.39it/s] [learner-step] step=2066, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028613034635782242}
learner:   0%|                                        | 2067/1000000 [07:40<43:23:45,  6.39it/s] [learner-step] step=2067, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029080858454108238}
learner:   0%|                                        | 2068/1000000 [07:40<43:08:03,  6.43it/s] [learner-step] step=2068, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02838590182363987}
learner:   0%|                                        | 2069/1000000 [07:40<42:34:07,  6.51it/s] [learner-step] step=2069, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02825929969549179}
learner:   0%|                                        | 2070/1000000 [07:41<42:49:26,  6.47it/s] [learner-step] step=2070, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029498634859919548}
learner:   0%|                                        | 2071/1000000 [07:41<42:49:22,  6.47it/s] [learner-step] step=2071, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02995196543633938}
learner:   0%|                                        | 2072/1000000 [07:41<43:25:15,  6.38it/s] [learner-step] step=2072, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029000798240303993}
learner:   0%|                                        | 2073/1000000 [07:41<43:38:35,  6.35it/s] [learner-step] step=2073, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02839074470102787}
learner:   0%|                                        | 2074/1000000 [07:41<43:52:21,  6.32it/s] [learner-step] step=2074, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028339864686131477}
learner:   0%|                                        | 2075/1000000 [07:41<44:28:31,  6.23it/s] [learner-step] step=2075, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027890460565686226}
learner:   0%|                                        | 2076/1000000 [07:42<44:41:12,  6.20it/s] [learner-step] step=2076, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029175039380788803}
learner:   0%|                                        | 2077/1000000 [07:42<44:31:55,  6.22it/s] [learner-step] step=2077, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02927575819194317}
learner:   0%|                                        | 2078/1000000 [07:42<43:52:47,  6.32it/s] [learner-step] step=2078, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029660018160939217}
learner:   0%|                                        | 2079/1000000 [07:42<44:14:52,  6.26it/s] [learner-step] step=2079, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028998883441090584}
learner:   0%|                                        | 2080/1000000 [07:42<43:57:48,  6.31it/s] [learner-step] step=2080, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029104992747306824}
learner:   0%|                                        | 2081/1000000 [07:42<43:35:11,  6.36it/s] [learner-step] step=2081, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027637509629130363}
learner:   0%|                                        | 2082/1000000 [07:42<43:11:47,  6.42it/s] [learner-step] step=2082, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027966756373643875}
learner:   0%|                                        | 2083/1000000 [07:43<44:10:02,  6.28it/s] [learner-step] step=2083, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.025519559159874916}
learner:   0%|                                        | 2084/1000000 [07:43<44:32:48,  6.22it/s] [learner-step] step=2084, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028932439163327217}
learner:   0%|                                        | 2085/1000000 [07:43<45:19:11,  6.12it/s] [learner-step] step=2085, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027525251731276512}
learner:   0%|                                        | 2086/1000000 [07:43<45:03:08,  6.15it/s] [learner-step] step=2086, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028393834829330444}
learner:   0%|                                        | 2087/1000000 [07:43<45:24:33,  6.10it/s] [learner-step] step=2087, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028137773275375366}
learner:   0%|                                        | 2088/1000000 [07:43<45:39:34,  6.07it/s] [learner-step] step=2088, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029576249420642853}
learner:   0%|                                        | 2089/1000000 [07:44<45:28:20,  6.10it/s] [learner-step] step=2089, publish_count=60, replay_buffer=3593, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02850351296365261}
learner:   0%|                                        | 2090/1000000 [07:44<45:51:54,  6.04it/s] [learner-step] step=2090, publish_count=60, replay_buffer=3594, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02833019755780697}
learner:   0%|                                        | 2091/1000000 [07:44<46:10:14,  6.00it/s] [learner-step] step=2091, publish_count=60, replay_buffer=3594, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029266392812132835}
learner:   0%|                                        | 2092/1000000 [07:44<46:08:03,  6.01it/s] [learner-step] step=2092, publish_count=60, replay_buffer=3594, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029544446617364883}
learner:   0%|                                        | 2093/1000000 [07:44<45:54:51,  6.04it/s] [learner-step] step=2093, publish_count=60, replay_buffer=3594, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.030659446492791176}
learner:   0%|                                        | 2094/1000000 [07:44<45:37:52,  6.07it/s] [learner-step] step=2094, publish_count=60, replay_buffer=3602, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028218520805239677}
learner:   0%|                                        | 2095/1000000 [07:45<45:56:57,  6.03it/s] [learner-step] step=2095, publish_count=60, replay_buffer=3602, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.026897720992565155}
learner:   0%|                                        | 2096/1000000 [07:45<45:51:38,  6.04it/s] [learner-step] step=2096, publish_count=60, replay_buffer=3602, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.0279696024954319}
learner:   0%|                                        | 2097/1000000 [07:45<45:32:44,  6.09it/s] [learner-step] step=2097, publish_count=60, replay_buffer=3602, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029337674379348755}
learner:   0%|                                        | 2098/1000000 [07:45<45:39:39,  6.07it/s] [learner-step] step=2098, publish_count=60, replay_buffer=3610, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02768401987850666}
learner:   0%|                                        | 2099/1000000 [07:45<46:09:38,  6.00it/s] [learner-step] step=2099, publish_count=60, replay_buffer=3610, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029725905507802963}
learner:   0%|                                        | 2100/1000000 [07:45<46:00:33,  6.02it/s] [learner-network-publish] reason=train_periodic_update, step=2100, cost=0.133s, leafs=81, total_params=8001041, checksum=8.340601, abs_mean=0.170578, sample_std=0.345458, head=[0.000317, -0.000397, 0.001241, 0.00039, -0.002161, 0.000802], shapes=[(6,), (256, 6), (6,), (256, 6)]
 [learner-step] step=2100, publish_count=61, replay_buffer=3610, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027954887598752975}
learner:   0%|                                        | 2101/1000000 [07:46<55:09:20,  5.03it/s] [learner-step] step=2101, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027510641142725945}
learner:   0%|                                        | 2102/1000000 [07:46<53:59:37,  5.13it/s] [learner-step] step=2102, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.028759419918060303}
learner:   0%|                                        | 2103/1000000 [07:46<51:03:11,  5.43it/s] [learner-step] step=2103, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027888810262084007}
learner:   0%|                                        | 2104/1000000 [07:46<48:53:28,  5.67it/s] [learner-step] step=2104, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02805274911224842}
learner:   0%|                                        | 2105/1000000 [07:46<48:03:21,  5.77it/s] [learner-step] step=2105, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.029156602919101715}
learner:   0%|                                        | 2106/1000000 [07:47<46:30:36,  5.96it/s] [learner-step] step=2106, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.027720587328076363}
learner:   0%|                                        | 2107/1000000 [07:47<46:09:12,  6.01it/s] [learner-step] step=2107, publish_count=61, replay_buffer=3619, demo_buffer=7166, critic_loss=N/A, actor_loss=N/A, temperature={'temperature_loss': 0.02724396623671055}
learner:   0%|                                        | 2108/1000000 [07:47<45:53:05,  6.04it/s]learner:   0%|                                        | 2108/1000000 [07:47<61:27:34,  4.51it/s]




###
###
###


##############################actor和learner输出分析：




这份 actor + learner 输出整体是正常的。重点结论：actor 和 learner 通信正常，网络参数正常下发，actor buffer 正在被 learner 接收，checkpoint 正常保存，episode 统计能正确同步到 learner；没有看到训练链路断开、网络不同步、reset 卡死、buffer 不增长这类问题。

1. actor 端状态：正常

actor 端先完成了一次单臂 reset：

✅ 单臂复位完成
pose=[0.2179, -0.3263, -0.1500, 0, 0, 0, 1]
gripper=80.0

这说明 reset 后夹爪是张开状态，位置也回到了预设安全起点。后面提示你按 VR Mode 0 重新接管，这个流程是对的。

actor 后续出现了多次：

[actor-network-recv] recv_count=39 / 40 / 41 / 42
total_params=8001041
checksum=...

这说明 actor 确实在持续收到 learner 发布的新网络参数。total_params=8001041 一直一致，说明模型结构没有乱；checksum 每次有轻微变化，说明 learner 真的在训练并更新参数，不是一直发同一个初始网络。

actor 端速度大概在：

11 ~ 13 it/s

比如：

3270 step 附近: 11.75 it/s
3368 step 附近: 12.60 it/s
3470 step 附近: 13.67 it/s

这个 actor 速度是可以的，没有明显卡死。

2. actor 的 VR / 脚本模式切换正常

输出里能看到：

🎮 [VR接管] 检测到 control_mode=0，发送 use_vr_mode=True
🎯 [VR归一化] 已设置当前手柄位姿为零点
✅ 模式切换返回: success=True

这说明 VR 接管时，系统会重新设置当前手柄位姿为零点，这正是你之前为了 VR 源头归一化要做的事情。

后面又有：

🤖 [脚本接管] 检测到 control_mode=2，发送 use_vr_mode=False
✅ 模式切换返回: success=True

这说明 reset / 脚本控制切换也正常。没有看到 mode 切换失败。

3. actor 第 12 个 episode 是一次失败，但不是 bug

actor 端有一条关键输出：

[actor-episode-end] episode=12, step=3580,
return=0.0000,
env_return=0.0000,
length=400,
duration=31.459,
success=0.0,
intervention_count=2,
intervention_steps=34,
recv_count=41,
applied_count=35

这说明第 12 个 episode 跑满了 400 步，没有成功，所以 return=0、success=0。这个和你 W&B 里看到的失败 episode 是一致的。

这不是数据 bug，而是一次正常的失败轨迹：

length=400 -> 达到最大步数
return=0   -> 没成功
success=0  -> classifier / env 没判成功

而且 episode 结束后 actor 立刻进入 reset 流程：

正在准备复位
切到脚本控制模式
开始向底层发送复位坐标
单臂复位完成

说明失败后 reset 逻辑也是通的。

4. actor 收网络的频率和 learner 发布频率是匹配的

actor 端：

recv_count=39, since_prev=9.226
recv_count=40, since_prev=7.744
recv_count=41, since_prev=8.874
recv_count=42, since_prev=9.587

learner 端在：

step=2000 publish_count=59
step=2050 publish_count=60
step=2100 publish_count=61

learner 大概每 50 learner step 发布一次网络。learner 速度约 6 it/s，所以每次发布间隔大概：

50 / 6 ≈ 8.3 秒

actor 收到网络的间隔是 7.7 ~ 9.6 秒，非常匹配。说明 learner 发布 -> actor 接收 这条链路是正常的。

更关键的是 checksum 对得上：

learner step=2000 发布：

checksum=8.343003

actor 随后 recv_count=41 收到：

checksum=8.343003

learner step=2050 发布：

checksum=8.342869

actor 后面 recv_count=42 收到：

checksum=8.342869

这说明 actor 收到的确实是 learner 发布的那份网络，不是旧网络、不是什么错位网络。

5. learner 端正在正常训练

learner 从 step 1980 到 2108 持续更新，速度大概：

5.8 ~ 6.4 it/s

这是正常的。中间 checkpoint 保存时速度短暂下降，后面又恢复了。

learner 端 buffer 数量也在增长：

step=1980: replay_buffer=3419, demo_buffer=7131
step=2000: replay_buffer=3473, demo_buffer=7131
step=2047: replay_buffer=3584, demo_buffer=7159
step=2100: replay_buffer=3610, demo_buffer=7166

这说明 learner 正在持续收到 actor 发来的 transition。replay_buffer 和 demo_buffer 都在增加，不是卡在固定数量。

这个非常重要：actor 不只是自己在跑，数据确实传到了 learner。

6. learner checkpoint 保存正常

learner 在 step=2000 保存 checkpoint：

Saving checkpoint at step: 2000
Finished saving checkpoint
[learner-checkpoint-save] step=2000, cost=1.966s

这说明 checkpoint 正常写入：

./rlpd_checkpoints_single/checkpoint_2000

没有看到 Orbax 保存失败、权限错误、磁盘错误。中间那些 Orbax 日志，比如：

No metadata found...
If checkpoint does not contain jax.Array then it is expected

这种不是致命错误，后面已经 Finished saving checkpoint，所以保存成功。

保存期间 learner 速度掉到：

1.25 it/s

然后逐步恢复到 6 it/s 左右，这是正常现象。checkpoint 保存会短暂阻塞训练。

7. learner-env-stats 和 actor episode-end 完全对应

actor 端 episode 12：

return=0.0000
length=400
duration=31.459
success=0.0
intervention_count=2
intervention_steps=34

learner 端对应：

[learner-env-stats] step=2047,
return=0.0000,
length=400,
duration=31.459,
success=0.0,
intervention_count=2,
intervention_steps=34

两边完全一致。

这说明 actor 的 episode 统计信息已经成功传给 learner，并且 learner 也正确记录了环境统计。这个链路是正常的。

8. learner 终端里 critic_loss=N/A, actor_loss=N/A 不是训练没跑

你 learner 输出里一直显示：

critic_loss=N/A
actor_loss=N/A
temperature={'temperature_loss': ...}

这个不用太紧张。因为你前面导出的 W&B history 里已经明确有：

actor/actor_loss
critic/critic_loss
grasp_critic/grasp_critic_loss

而且这些 loss 有正常数值和下降趋势。

所以这里的 N/A 更像是 终端打印逻辑没有正确从 update_info 里取到 actor_loss / critic_loss，不是训练没发生。真正的 loss 已经进了 W&B。

也就是说：

训练是正常的；
W&B 有 loss；
只是 learner 终端这行简化打印没有显示出来。

如果你后面想让终端也打印 loss，需要改 train_rlpd.py 里 [learner-step] 那段取 key 的逻辑，但这不是训练正确性的核心问题。

9. temperature_loss 稳定

learner 端 temperature loss 大概在：

0.027 ~ 0.031

比如：

0.0284
0.0300
0.0291
0.0279

没有 NaN，没有爆炸。这和你 W&B 里 temperature 正常下降是一致的。

10. 目前唯一需要关注的是：失败 episode 和人工介入

当前 actor 第 12 个 episode 失败了：

return=0
success=0
length=400
intervention_count=2
intervention_steps=34

这说明策略当前还没有稳定到每次都成功，或者这次人工介入不足 / 操作没完成。这个阶段是正常的，因为你还在早期 RLPD。

后续重点不是看单次失败，而是看趋势：

success rate 是否上升
episode return 是否稳定为 1
intervention_count 是否下降
intervention_steps 是否下降
失败的 400 步超时是否越来越少

如果训练更久以后仍然大量 length=400, return=0，那再分析策略、reward classifier、动作尺度或干预方式。

最终结论

这份 actor / learner 输出是正常的，核心链路都通了：

✅ actor reset 正常
✅ VR / 脚本模式切换正常
✅ actor 持续收到 learner 网络
✅ learner 发布频率和 actor 接收频率匹配
✅ checksum 对齐，说明 actor 收到的是正确网络
✅ replay_buffer / demo_buffer 持续增长
✅ learner 正在训练
✅ checkpoint_2000 保存成功
✅ actor episode-end 能同步到 learner-env-stats
✅ W&B loss 记录正常，只是终端 loss 打印是 N/A

一句话：这次输出没有显示系统性错误；训练链路、通信链路、buffer 链路、checkpoint 链路都是通的。现在主要看更长时间训练后，成功率和人工介入是否持续改善。


###
###
###
###
##
###
###
###


#####################wandb损失分析：


可以。根据你导出的 W&B history CSV，这次训练虽然只跑到 _step=2100，但已经能看出很多东西：W&B 记录正常，loss 正常下降，critic / grasp critic 都已经从初始大误差快速稳定，环境端也记录到了 12 个 episode，其中 10 个成功、2 个失败。整体训练链路是通的。

1. W&B 数据记录是完整的

你导出的 history 有：

history rows: 443
_step: 0 -> 2100
metric 更新行数: 211
episode 记录行数: 12

也就是说，这个 run 虽然被你手动停止，状态显示 killed，但它已经正常记录了训练指标。列名里有这些核心指标：

actor/actor_loss
actor/entropy
actor/temperature
critic/critic_loss
critic/predicted_qs
critic/target_qs
grasp_critic/grasp_critic_loss
grasp_critic/predicted_grasp_qs
grasp_critic/target_grasp_qs
temperature/temperature_loss
environment/episode/r
environment/success
environment/grasp_penalty

所以 W&B 已经能正常看训练 loss，不只是日志。你网页上应该能看到这些曲线。

2. critic loss：非常健康，从大误差快速降到很小

critic/critic_loss：

first: 18.8146
last: 0.00133
max: 18.8146
median: 0.00323
前 10 个均值: 1.9285
后 10 个均值: 0.00224

这个非常好。开始很大，是因为刚启动时 Q 网络还没有拟合；很快下降到 0.01 以下，后期基本在 0.001 ~ 0.01 附近波动。

这说明：

critic 已经能很好拟合当前 replay / demo batch 里的 target Q
没有出现 critic loss 爆炸
没有出现 NaN
没有出现长期大幅震荡

从这点看，learner 的 critic 更新是正常的。

3. critic Q 值：predicted_qs 和 target_qs 已经对齐

critic/predicted_qs：

first: -5.1822
last: -0.5617
后 10 个均值: -0.5315

critic/target_qs：

first: -0.9102
last: -0.5565
后 10 个均值: -0.5331

开始时 predicted Q 和 target Q 差距很大：

predicted_qs ≈ -5.18
target_qs    ≈ -0.91

所以一开始 critic loss 很大。后期两者基本对齐：

predicted_qs ≈ -0.53
target_qs    ≈ -0.53

这就是 critic loss 降下来的直接原因。这说明 critic 没有乱学，Q 估计已经和 target 基本一致。

不过也要注意：Q 值整体是负的，这不是一定有问题，因为你的 reward 里有 sparse success reward，同时还有 temperature / entropy / bootstrap 等项；如果训练继续，Q 可能还会随着策略和数据分布变化继续调整。现在最重要的是它稳定、对齐、不爆炸。

4. actor loss：明显下降，策略更新正常

actor/actor_loss：

first: 5.1978
last: 0.5314
min: 0.4445
median: 0.8065
前 10 个均值: 1.4157
后 10 个均值: 0.5018

这个趋势也不错。它从启动时的高值快速下降，后面稳定在 0.5 左右。

这说明：

actor 正在根据 critic 反馈更新
没有出现 actor loss 持续升高
没有出现异常爆炸

但 actor loss 本身不是“越低越一定好”的绝对指标，后续还要结合环境成功率、动作稳定性和 Q 值看。当前只能说：actor 更新是稳定的。

5. entropy 和 temperature：探索温度在下降，符合 SAC 类训练现象

actor/entropy：

first: -2.1147
max: 3.8995
last: 2.5016
后 10 个均值: 2.7371

actor/temperature：

first: 0.0100
last: 0.00459
前 10 个均值: 0.00980
后 10 个均值: 0.00466

temperature 从 0.01 降到 0.0046，说明自动温度项正在降低探索权重。entropy 后期仍在 2.5 ~ 3.0 左右，不是塌缩到极低值。

这个比较健康：

temperature 没有爆炸
entropy 没有塌缩
策略还保留一定随机性

如果后续训练很久后策略仍然动作很抖，可以再看 entropy 是否偏高；如果策略完全不探索，就看 entropy 是否过低。当前这个短 run 还不能下定论。

6. grasp critic：也已经从巨大初始误差降到稳定

grasp_critic/grasp_critic_loss：

first: 65.9795
last: 0.00252
median: 0.00237
前 10 个均值: 6.7065
后 10 个均值: 0.00228

这非常关键。它说明你的 learned-gripper / grasp critic 分支确实在训练，而且已经快速稳定。

grasp_critic/predicted_grasp_qs：

first: 8.7223
last: 0.2620
后 10 个均值: 0.2261

grasp_critic/target_grasp_qs：

first: 0.6171
last: 0.2327
后 10 个均值: 0.2236

后期 predicted grasp Q 和 target grasp Q 很接近：

predicted_grasp_qs ≈ 0.226
target_grasp_qs    ≈ 0.224

这说明 grasp critic 也已经对齐。结合你之前 buffer inspect 里 grasp_penalty 已经完全正确，这里可以判断：夹爪惩罚链路和 grasp critic 训练链路都是通的。

7. grasp_rewards 数值符合预期

grasp_critic/grasp_rewards：

mean: 0.00396
min: -0.0003125
max: 0.015625
last: -0.000156

这里不是每条都等于 -0.02，因为这是 batch 平均后的 reward/penalty 统计。你 buffer 里夹爪惩罚是稀疏事件，batch 里大多数 transition 是 0，少数是 -0.02，所以平均值会很小，接近 0 是正常的。

这点和之前 buffer inspect 一致：非零 grasp penalty 很少，只在 close/open 事件上出现。

8. 环境 episode：12 个 episode，10 个成功，2 个失败

W&B 记录了 12 条 episode 级别数据：

成功 episode: 10
失败 episode: 2
episode/r 总和: 10

成功率约：

10 / 12 = 83.3%

失败的两个 episode 是：

_step=1067, length=400, reward=0, intervention_count=3, intervention_steps=186
_step=2047, length=400, reward=0, intervention_count=2, intervention_steps=34

它们都长度到 400，很像达到 max episode steps 后失败截断。

成功 episode 的长度大概是：

127, 188, 190, 191, 203, 267, 290, 319, 320, 354

失败 episode 长度都是：

400

所以环境端逻辑也符合预期：

成功 -> reward=1
失败/超时 -> reward=0

这次 12 个 episode 里有 10 个成功，说明在这段短训练/采样中，任务并不是完全失败；但这不能直接说明策略已经学会，因为里面有人工介入。要看真实策略效果，需要后续看 intervention_count 是否下降、无介入成功率是否上升。

9. intervention 数据：还不能说明完全自动成功

成功 episode 里 intervention_count 大多是 1~2，失败 episode 也有 2~3。例如：

成功: intervention_count=1 或 2
失败: intervention_count=2 或 3

这说明当前收集的数据仍然包含人工介入。现在更像是 RLPD/HIL 阶段正在用人工介入收集成功数据并训练，不是纯策略已经独立完成任务。

后续你要重点看：

environment/episode/intervention_count 是否逐渐下降
environment/episode/intervention_steps 是否逐渐下降
environment/episode/r 是否保持为 1
environment/success 是否保持上升或稳定

理想训练后期应该是：

成功率高
intervention_count 下降
intervention_steps 下降
episode length 下降或稳定
10. timer：训练速度整体正常，但偶尔有尖峰

大部分时间：

timer/step_env median ≈ 0.0697 s
timer/total median ≈ 0.0724 s
timer/train median ≈ 0.0622 s
timer/train_critics median ≈ 0.0440 s

这说明主循环大多数时候是正常的。

但是有明显尖峰：

timer/train max ≈ 32.18 s
timer/train_critics max ≈ 27.65 s
timer/step_env max ≈ 2.04 s
timer/total max ≈ 2.57 s

这些通常来自：

第一次 JAX 编译
网络首次更新
保存/同步
actor/learner 通信等待
系统调度

如果只是开头或偶发，不是问题。你的 p95 基本还可以：

timer/step_env p95 ≈ 0.128 s
timer/total p95 ≈ 0.153 s
timer/train p95 ≈ 0.071 s

说明大部分时间没有严重卡顿。

11. 目前最重要的结论

这次短 run 的训练曲线整体是健康的：

✅ W&B metric 正常记录
✅ critic_loss 从 18.8 降到 0.0013
✅ predicted_qs 和 target_qs 已经对齐
✅ actor_loss 从 5.2 降到 0.53
✅ temperature 从 0.01 降到 0.0046
✅ entropy 没有塌缩
✅ grasp_critic_loss 从 66 降到 0.0025
✅ predicted_grasp_qs 和 target_grasp_qs 已经对齐
✅ episode 记录正常
✅ 12 个 episode 中 10 个成功
✅ 失败 episode 都是 400 步超时，符合失败截断现象
12. 但现在还不能判断“策略已经学会”

因为这个 run 很短，只到 _step=2100，而且 episode 成功里仍有人工介入。当前能判断的是：

训练代码正常
数据链路正常
loss 没有爆炸
critic / grasp critic 正常收敛到当前 batch
W&B 正常记录

还不能判断：

策略是否已经能独立完成任务
是否真正收敛
是否会过拟合 demos
是否 intervention 会逐渐减少

后续你发网页图表时，我最想看这些曲线：

environment/episode/r
environment/success 或 environment/is_success
environment/episode/intervention_count
environment/episode/intervention_steps
critic/critic_loss
actor/actor_loss
critic/predicted_qs 和 critic/target_qs
grasp_critic/grasp_critic_loss
grasp_critic/predicted_grasp_qs 和 grasp_critic/target_grasp_qs
actor/temperature
actor/entropy
最终判断

这次训练的早期表现是正常且健康的。
loss 已经快速稳定，critic 和 grasp critic 都没有异常，reward / success / intervention 也被正常记录。现在可以继续正式训练更长时间，然后重点观察：成功率是否持续高、人工介入是否减少、策略是否能越来越少依赖 VR 接管

###
###
###



###############################loss分析（https://wandb.ai/erenjaeger-hit/hil-serl/workspace?nw=nwusererenjaeger）

结论先说：这组 W&B 图整体是健康的，训练链路是通的，loss 没有爆炸，critic / grasp critic 都已经快速对齐 target，actor 也在稳定更新。 但它只跑到大约 2.1k step，时间还短，所以现在能判断“训练正常”，还不能判断“策略已经真正学会并能脱离人工介入”。

1. actor：actor_loss、entropy、temperature
actor/temperature

这条曲线从大约：

0.010 -> 0.0046

一路平滑下降。CSV 里对应统计也是：初始约 0.0100，最后约 0.00459，后 10 个点均值约 0.00466。

这个含义是：SAC 的自动温度系数在下降，策略探索权重逐渐变小。这一般是正常现象。刚开始策略比较随机，需要更多熵探索；训练一段后，critic 变稳定，actor 开始更偏向利用已学到的 Q 值，temperature 就下降。

这条曲线非常平滑，没有突然爆炸、没有 NaN、没有震荡到很大值，所以 temperature 是健康的。

actor/entropy

这条曲线一开始有一个很低的点，随后快速上升到 3 左右，后面大致在：

2.5 ~ 3.8

之间波动。CSV 里 actor/entropy 最大约 3.899，最后约 2.502，后 10 个点均值约 2.737。

它的含义是：策略动作分布的随机性 / 探索程度。

现在这个状态比较合理：

没有掉到接近 0  -> 策略没有完全塌缩成确定动作
没有无限升高     -> 策略也没有变成完全随机

后期如果你发现机器人动作很抖、很随机，就要看 entropy 是否长期偏高；如果机器人完全不探索、动作很死，就看 entropy 是否过低。现在这条曲线没问题。

actor/actor_loss

图上最开始有一个高点，大约 5，然后快速掉到 1 附近，后面逐渐下降到 0.5 左右。CSV 里是：初始 5.1978，最后 0.5314，后 10 个点均值约 0.5018。

这说明 actor 正在从 critic 学到的 Q 值里更新策略，并且更新没有发散。

不过 actor loss 不能简单理解成“越低越好”。在 SAC/RLPD 里，它和 Q 值、温度、熵项都有关系。你现在应该这样看：

actor_loss 从很高降下来      -> 正常
actor_loss 后期比较平稳      -> 正常
没有持续爆炸、没有 NaN       -> 正常

所以 actor 部分总体判断是：策略更新正常，没有异常。

2. Charts：四个 learning rate

你这里有：

temperature_lr = 0.0003
grasp_critic_lr = 0.0003
critic_lr = 0.0003
actor_lr = 0.0003

四条都是水平直线。CSV 里这些 learning rate 全部恒定为约 3e-4。

这说明你的训练没有使用学习率衰减 scheduler，或者当前配置下 scheduler 没有变化。这个不是问题。对 SERL / SAC 类训练来说，3e-4 是常见量级。

它的含义是：

actor 更新步长固定
critic 更新步长固定
grasp critic 更新步长固定
temperature 更新步长固定

从目前 loss 曲线看，这个学习率没有导致发散，所以可以继续用。

3. critic：target_qs、predicted_qs、rewards、critic_loss
critic/critic_loss

这是最关键的 critic 训练曲线。图上最开始有一个巨大 spike，大约 18，之后迅速贴近 0。CSV 里初始 18.8146，最后 0.00133，中位数只有 0.00323，后 10 个点均值约 0.00224。

这个非常健康。含义是：

初始 Q 网络和 target Q 差很大
训练几十步后 Bellman error 快速下降
后期 predicted Q 已经能拟合 target Q

这个 spike 不是 bug。刚开始网络随机初始化，Q 值差很大，critic loss 大是正常的。关键是它快速降下来了，没有继续爆炸。

critic/predicted_qs

图上 predicted_qs 从很低的负值逐渐上升。CSV 里初始是约：

-5.18

最后到约：

-0.56

后期和 target_qs 非常接近。

这说明刚开始 critic 对当前动作的 Q 估计非常悲观，后来逐步修正到合理范围。

critic/target_qs

图上 target_qs 大概从 -1.1 附近逐渐上升到 -0.55 附近。CSV 里 target_qs 初始约 -0.91，最低到过 -1.10，最后约 -0.556。

关键是后期：

predicted_qs ≈ -0.56
target_qs    ≈ -0.56

这说明 critic 的预测已经追上 target。这个和 critic_loss 降到很低是完全一致的。

critic/rewards

图上是很多稀疏尖峰，幅度大概：

0 ~ 0.015625

CSV 里 critic/rewards 最大就是 0.015625，中位数约 0.00390625。

这个不是单条 transition 的 reward，而是 训练 batch 里的 reward 平均值。你的任务 reward 是 sparse reward，单条成功 reward 是 1，大部分 transition reward 是 0。如果 batch size 是 256，那么：

1 / 256 = 0.00390625
4 / 256 = 0.015625

所以这些尖峰的含义是：当前 batch 里抽到了若干成功 transition。这非常符合 sparse reward 任务。

critic 部分最终判断：critic 训练非常正常，predicted Q 和 target Q 对齐，loss 快速下降，没有发散。

4. environment：grasp_penalty、episode/t、episode/r、episode/l、intervention

这部分是判断真实任务表现的关键。

environment/episode/r

图上 episode reward 只有 0 和 1。CSV 里一共记录了 12 个 episode，其中：

成功: 10 个
失败: 2 个
平均 reward: 0.8333

也就是这个短 run 中，episode 成功率大约：

10 / 12 = 83.3%

但要注意：这不是纯策略成功率，因为还有人工 intervention。现在只能说明：环境 reward 记录正常，成功/失败标记正常，训练过程中确实采到了成功 episode。

两个失败 episode 都是：

reward=0
length=400
success=False

这很像达到最大步数后超时失败。

environment/episode/l

这条是 episode length，图上范围大概：

127 ~ 400

CSV 里最短 127，最长 400，平均约 270.75。

含义是：

成功 episode 通常 127~354 步结束
失败 episode 是 400 步超时

所以 episode length 的大幅上升不一定是坏事，它通常代表这次 episode 没有及时成功，最后跑满 max_episode_steps。

你图里有两次接近 400 的点，正好对应那两个失败 episode。

environment/episode/t

这个是每个 episode 用时。图上大概在：

15s ~ 41s

CSV 里最短约 15.33s，最长约 41.35s，平均约 24.68s。

它和 episode length 大体对应。失败 episode 长度 400，所以时间更长；成功 episode 较短，所以时间一般更短。

这个用来判断执行速度是否异常。现在看：

没有持续变慢
没有 episode 时间突然变成几分钟

所以环境执行速度基本正常。

environment/episode/intervention_count

图上范围是：

1 ~ 3

CSV 里最小 1，最大 3，平均约 1.58。

含义是：每个 episode 人工接管了几段。现在每个 episode 仍然有 1 到 3 次人工介入，所以这不是纯自动成功。

这个指标后续非常重要。理想训练趋势是：

intervention_count 逐渐下降
episode/r 仍然保持 1

也就是说，机器人越来越少需要你接管，但仍然能成功。

environment/episode/intervention_steps

图上大部分在：

100 ~ 200

最后一个失败 episode 掉到大约 34。CSV 里 intervention_steps 最小 34，最大 224，平均约 161.25。

这个含义是：人工介入持续了多少步。

目前这个值还比较高，说明训练早期仍然大量依赖人工。尤其成功 episode 里也有很多 100+ intervention steps，所以现在还不能说策略已经独立完成任务。

后续你要看：

intervention_steps 是否从 150~200 逐渐下降到几十，甚至接近 0

如果 reward 还是 1，而 intervention_steps 下降，那才说明策略真的学会了更多。

environment/grasp_penalty

图上大部分是：

-0.02

最后有一次到 0。CSV 里 12 个 episode 中，大部分 episode 的 environment/grasp_penalty=-0.02，最后一个失败 episode 是 0。

这里要注意：这个图不是 buffer 里每条 transition 的 grasp penalty 分布，而是 W&B 记录的 episode/environment 层面的值。它表示该 episode 里最后记录到的或汇总后的 grasp penalty 指标。

结合你之前 inspect buffer 的结果，真正关键的是：

close/open 事件 -> -0.02
hold -> 0

这一点你 buffer inspect 已经验证过是正确的。所以这个图里大部分 episode 出现 -0.02，说明这些 episode 中发生过一次夹爪事件；最后一次为 0，可能是那次失败 episode 没有发生有效 close/open 事件，或者最后记录的是 0。

这个不异常。

5. grasp_critic：target_grasp_qs、predicted_grasp_qs、grasp_rewards、grasp_critic_loss
grasp_critic/grasp_critic_loss

图上最开始有一个非常高的 spike，大约 66，然后迅速下降到接近 0。CSV 里初始是 65.98，最后是 0.00252，中位数约 0.00237。

这和 critic_loss 非常类似：初始网络完全没拟合，误差大；训练很快把 predicted grasp Q 拉到 target grasp Q 附近。

这说明 grasp critic 分支确实在训练，而且没有发散。

grasp_critic/predicted_grasp_qs

图上最开始有一个大 spike，大概 8.7，很快掉下来，后面缓慢上升到 0.2 左右。CSV 里初始 8.72，最后 0.262，后 10 个点均值约 0.226。

这说明初始 grasp Q 预测完全不准，后面逐渐和 target 对齐。

grasp_critic/target_grasp_qs

图上前期有负值波动，后来逐渐上升到 0.2 左右。CSV 里最后约 0.233，后 10 个点均值约 0.224。

后期：

predicted_grasp_qs ≈ 0.226
target_grasp_qs    ≈ 0.224

这说明 grasp critic 已经对齐 target。这个非常重要，因为你前面花了很多时间修正 grasp_penalty，现在 W&B 曲线说明：grasp critic 确实收到了有效信号，并且能拟合。

grasp_critic/grasp_rewards

图上也是稀疏尖峰，范围大概：

-0.0003 ~ 0.015625

CSV 里最小 -0.0003125，最大 0.015625。

这个很关键。它说明 batch 里既有正 reward，也有非常小的负 penalty 平均值。

为什么 -0.02 会变成 -0.0003125？因为这是 batch 平均。假设 grasp critic batch size 是 64，那么：

-0.02 / 64 = -0.0003125

这正好对应你看到的最小值。说明夹爪 penalty 的数值进入了 grasp critic 训练统计，而且是按 batch 平均显示的。

为什么最大是 0.015625？如果 batch size 是 64：

1 / 64 = 0.015625

说明某些 batch 里抽到了成功 reward。

所以这条曲线非常符合你的数据设计：

大多数 transition reward/penalty 是 0
少数成功 transition 给正 reward
少数夹爪事件给 -0.02 penalty
W&B 显示的是 batch mean，所以幅度很小
6. temperature：temperature_loss

temperature/temperature_loss 图上从大约 0.06+ 逐渐下降到 0.03 左右。CSV 里最大约 0.0673，最后约 0.02795，后 10 个点均值约 0.02893。

它的含义是自动温度调节器的优化损失。它下降说明 temperature 在朝目标 entropy 调整。

配合 actor/temperature 曲线一起看：

temperature_loss 下降
actor/temperature 下降
entropy 没有塌缩

这是健康的。

7. timer：训练耗时、环境耗时、采样耗时
timer/train_critics

图上最开始有一个巨大 spike，大概 27s，之后接近 0。CSV 里初始约 27.65s，后 10 个点均值约 0.0449s。

这个巨大 spike 基本可以判断是 JAX 首次编译。第一次 train critic 时，JAX/XLA 需要编译计算图，所以会非常慢。后面每步只有几十毫秒，这是正常的。

timer/train

图上同样开头有大 spike，大约 32s，后面稳定。CSV 里初始 32.18s，后 10 个点均值约 0.0628s。

这也是第一次完整训练 step 编译造成的。不是性能问题。

timer/total

图上大多数点接近 0.07s，但有一些 spike 到 1~2.5s。CSV 里 median 约 0.072s，最大约 2.57s，p95 约 0.153s。

含义是每个 actor/env loop 或总 step 的耗时。大多数时候非常稳定，偶发 spike 可能来自：

reset
VR / 脚本模式切换
相机帧等待
reward classifier 推理卡顿
网络参数接收
系统调度

因为 p95 只有 0.153s，说明 spike 是少数，不是长期卡顿。

timer/step_env

图上也有周期性 spike，最高大约 2s。CSV 里 median 约 0.0697s，最大约 2.04s，p95 约 0.128s。

这说明 env.step 大多数时候大概：

0.07s / step

也就是大约 14Hz 左右，和 actor 终端里 11~13 it/s 接近。你之前 actor 输出也显示 actor 大约在 11~13 it/s，learner 约 5~6 it/s，网络发布和接收频率匹配。

偶发 2 秒 spike 不一定是问题，真实机器人环境里很常见。只要不是每一步都 2 秒，就可以接受。

timer/sample_replay_buffer

图上几乎是一条很低的线。CSV 里 median 约 0.026s，后 10 个点均值约 0.0261s。

这说明 replay buffer 采样很快，没有成为瓶颈。

timer/sample_actions

图上大多数接近 0，有两三个 spike 到 0.2s。CSV 里 median 约 0.0026s，最大约 0.202s。

含义是策略采样动作的时间。绝大多数时候非常快，偶发 spike 可能来自：

第一次 JIT
参数更新
GPU 调度
后台通信

不是大问题。

timer 总结：系统速度基本正常，主要慢点是首次编译和少量 env spike。

8. System：GPU fanSpeed

你截图里显示了多个 GPU fanSpeed：

gpu.5.fanSpeed: 约 42 -> 56
gpu.4.fanSpeed: 约 42 -> 90
gpu.3.fanSpeed: 约 45 -> 100
gpu.2.fanSpeed: 约 30 -> 75+
gpu.1.fanSpeed: 约 42 附近
gpu.0.fanSpeed: 约 41~42 附近波动

这个说明服务器 GPU 风扇在训练过程中明显升速，特别是 GPU2/3/4。它表示机器确实在承受 GPU 负载或温度上升。

但只看 fanSpeed 不能判断是否过热。你还需要看：

system/gpu.X.temp
system/gpu.X.memoryAllocated
system/gpu.X.gpuUtilization

如果风扇高，但温度稳定在合理范围，比如 70~80℃ 内，通常没问题。如果温度接近 90℃ 或以上，就需要关注散热。

目前从训练曲线看，没有出现：

训练速度持续下降
loss 变 NaN
进程 OOM
GPU 崩溃

所以风扇升高本身不是错误，只是说明 GPU 在工作。

9. actor / learner 通信与 W&B 图的对应关系

你前面贴的 actor/learner 输出能和 W&B 图对应上：

learner 在 step=2000、2050、2100 都发布网络，actor 端收到网络，参数数量 8001041 一致，checksum 对齐。比如 learner step=2050 发布 checksum 8.342869，actor 后面收到的 recv_count=42 也是 checksum 8.342869。

这说明：

learner 正在训练
learner 正在发布新网络
actor 正在收到新网络
actor 使用的不是旧策略

W&B 里的 actor_loss、critic_loss、temperature、Q 值变化，不是空跑出来的，确实对应 learner 更新过程。

10. 目前这轮训练最重要的判断
已经正常的部分
✅ actor_loss 正常下降
✅ critic_loss 快速下降到很低
✅ predicted_qs 和 target_qs 对齐
✅ grasp_critic_loss 快速下降到很低
✅ predicted_grasp_qs 和 target_grasp_qs 对齐
✅ temperature 平滑下降
✅ entropy 没有塌缩
✅ learning rate 稳定
✅ sparse reward batch mean 幅度符合预期
✅ grasp reward 中能看到 -0.0003125，说明 -0.02 penalty 进入 batch 平均
✅ episode reward 正常记录 0/1
✅ 成功和失败 episode 都能正确记录
✅ timer 没有长期卡顿
✅ actor/learner 网络同步正常
还不能下结论的部分
⚠️ 训练时间太短，只有约 2.1k step
⚠️ 成功 episode 仍然有大量人工 intervention
⚠️ 当前 10/12 成功率不能等价于纯策略成功率
⚠️ 还需要看 intervention_count / intervention_steps 是否随训练下降
11. 你后续最应该盯的几条曲线

后面继续训练时，最关键不是只看 loss，而是看这些组合：

组合 1：是否真的学会
environment/episode/r
environment/success
environment/episode/intervention_count
environment/episode/intervention_steps

理想趋势：

episode/r 保持接近 1
success 保持 True
intervention_count 下降
intervention_steps 下降

这才表示策略越来越独立。

组合 2：critic 是否稳定
critic/critic_loss
critic/predicted_qs
critic/target_qs

理想趋势：

critic_loss 不爆炸
predicted_qs 和 target_qs 不长期分离
Q 值不要突然大幅正/负爆炸

你现在是健康的。

组合 3：grasp critic 是否稳定
grasp_critic/grasp_critic_loss
grasp_critic/predicted_grasp_qs
grasp_critic/target_grasp_qs
grasp_critic/grasp_rewards

理想趋势：

grasp_critic_loss 低且稳定
predicted_grasp_qs 跟 target_grasp_qs 接近
grasp_rewards 偶尔有小负值，说明 penalty 还在进入训练

你现在也是健康的。

组合 4：策略是否过随机或过确定
actor/entropy
actor/temperature
temperature/temperature_loss

理想趋势：

temperature 平滑下降或稳定
entropy 不塌缩到 0
entropy 不长期异常高

你现在没有问题。

最终结论

这组图整体非常好，说明你现在的 RLPD 训练已经进入了“数据链路和训练链路都正常”的状态。

一句话总结：

critic 和 grasp critic 都已经稳定拟合 target；actor loss 和 temperature 都在正常下降；environment 能记录成功/失败和人工介入；timer 没有持续瓶颈；system 风扇上升说明 GPU 在工作但暂时看不出异常。当前最大的问题不是代码 bug，而是训练还太早，策略仍依赖人工介入。后续重点观察 intervention_count 和 intervention_steps 是否下降