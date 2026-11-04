import gtsam 
from gtsam import noiseModel, Rot3
from gtsam import Similarity3
from gtsam.symbol_shorthand import X
import numpy as np

class Optimiser: 
    def __init__(self):
        pass 
    
    def optimise_sim3(self, graph): 

        factor_graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values() 

        # relative_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3]*7))
        relative_loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*3 + [0.1]*3 + [0.1]))
        relative_odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*3 + [0.1]*3 + [0.1]))
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001]*7))
        huber = noiseModel.mEstimator.Huber(k=1.345)
        relative_noise_robust = noiseModel.Robust.Create(huber, relative_loop_noise)

        # Add room nodes
        for node in graph.room_nodes.values(): 
            idx = node.idx 

            pose_array = node.ref_pose
            pose = self._pose_to_sim3(pose_array)

            if idx == 0: 
                # Add prior on first pose 
                factor_graph.add(gtsam.PriorFactorSimilarity3(X(idx), pose, prior_noise))
            
            initial_values.insert(X(idx), pose)
        
        for (src, tgt), edges_list in graph.room_to_room_edge.items(): 
            # Skip if nodes were deleted from room_nodes but edges remain
            if src not in graph.room_nodes or tgt not in graph.room_nodes:
                continue
                
            for edge in edges_list:
                rel_transform = self._pose_to_sim3(edge.relative_transform)
                
                if edge.edge_type == "loop":
                    factor_graph.add(
                        gtsam.BetweenFactorSimilarity3(X(src), X(tgt), rel_transform, relative_noise_robust)
                    )
                else: 
                    factor_graph.add(
                        gtsam.BetweenFactorSimilarity3(X(src), X(tgt), rel_transform, relative_odom_noise)
                    )
        
        # initial_values.print("")
        # factor_graph.print("")
        parameters = gtsam.LevenbergMarquardtParams()
        # parameters.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_values, parameters)
        result = optimizer.optimize()


        # Update the graph
        for node in graph.room_nodes.values():
            key = X(node.idx)
            if result.exists(key):
                optimised_pose3 = result.atSimilarity3(key)
                node.optimised_ref_pose = self._sim3_to_pose(optimised_pose3)

    
    def _pose_to_sim3(self, pose, scale=1.0):
        rot_matrix = pose[:3, :3]
        translation = pose[:3, 3]
        rot3 = Rot3(rot_matrix)
        sim3 = Similarity3(rot3, translation, scale)
        return sim3
    
    def _sim3_to_pose(self, sim3): 
        rot_matrix = sim3.rotation().matrix()
        translation = sim3.translation()
        scale = sim3.scale()
        scaled_rot = scale * rot_matrix 
        pose = np.eye(4)
        pose[:3, :3] = scaled_rot
        pose[:3, 3] = translation 
        return pose

    
