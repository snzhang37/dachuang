
# 评判一个算法需要三个指标：
# 1. 时间
# 2. 成功率
import Date_processing
import estimate_upper_bound
import service_recovery_3_25
import service_recovery_version2
import cluster4
import GreedyRSA_SINGLE
def compare_method():

    args = {"distance_margin" : 1000,
            "ots_margin" : 10,
            "osnr_margin" : 0.01,
            "file_path" : "new_example/",
            "file_name" : "example_200",
            "band" : 8,
            "c_max" : 964,

            "FILE_ADD" : ""
            }
    args.update({"subgraph_file_path" : './subgraphs1/' + args["file_name"] + '/'})

    S1 = service_recovery_3_25.ServiceRecovery(**args)
    num_of_succeed, num_of_services, avg_time, resource_occupation = S1.run()
    print(num_of_succeed, num_of_services, avg_time, resource_occupation)
    G = Date_processing.create_topology(**args)
    cluster4.cluster(G=G, **args)
    S2 = service_recovery_version2.ServiceRecovery(**args)
    num_of_succeed, num_of_services, avg_time, resource_occupation = S2.run()
    print(num_of_succeed, num_of_services, avg_time, resource_occupation)
    S3 = GreedyRSA_SINGLE.GreedyRSA(**args)
    num_of_succeed, num_of_services, avg_time, resource_occupation = S3.RSA()
    print(num_of_succeed, num_of_services, avg_time, resource_occupation)



if __name__ == '__main__':
    compare_method()