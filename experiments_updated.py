import math, random
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import statistics
import pandas as pd
from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *
import warnings
warnings.filterwarnings("ignore")

""" Globals """
EXPERIMENT_COUNT = 10
RANGE_SIZE = 10
error_list_year = []

""" Helpers """

def read_dataset(filename):
    df = pd.read_csv(filename, sep=',', header = 0)
    return df

def get_domain(df): 
    max_bin = int(max(df)/RANGE_SIZE)
    domain = [*range(0,max_bin+1)]
    return np.array(domain)
    
def hash_func_with_modulo(domain,g):
    list1 = [([])*g for i in range(g)]
    select_list = list(range(len(domain)))
    counter = 0 
    while select_list != []:
        i = random.choice(select_list)
        select_list.remove(i)
        counter +=1
        if counter == g :
            counter = 0 
        list1[counter].append(i)
    return list1
    
def calculate_average_error(actual_hist, est_hist):
    mean_error = np.average(np.absolute(np.array(est_hist) - np.array(actual_hist)))
    return mean_error

# GRR

def perturb_grr(val, epsilon, domain):
    d = len(domain)
    p = (math.exp(epsilon)/(math.exp(epsilon) + d - 1))
    q = (1- p)/(d-1)
    prob_dist = [p if int(val/RANGE_SIZE) == x else q for x in domain ]
    prob_dist = np.array(prob_dist)
    prob_dist /= prob_dist.sum()
    return np.random.choice(domain, p=prob_dist)


def estimate_grr(perturbed_values, epsilon , domain):
    d = len(domain)
    p = (math.exp(epsilon)/(math.exp(epsilon) + d - 1))
    q = (1- p)/(d-1)
    Iv = Counter(perturbed_values)
    C_n = [0]*d
    for v in range(d):
        C_n[v]= (Iv[domain[v]] - (len(perturbed_values)*q))/(p-q)
    
    C_n = [0 if x<0 else x for x in C_n] # for negative values makes them 0
    C_n = [round(x) for x in C_n]
    return C_n


def grr_experiment(dataset, epsilon, domain):
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):      
        perturbed_list = [perturb_grr(val, epsilon, domain) for val in dataset]
        estimated_list = estimate_grr(perturbed_list, epsilon, domain)
        d = len(domain)
        actual_list = [0]*d  
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1
        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

# RAPPOR

def encode_rappor(val, domain):
    return [1 if (int(val/RANGE_SIZE))==x else 0 for x in domain]


def perturb_rappor(encoded_val, epsilon, domain):
    p = (math.exp(epsilon/2)/(math.exp(epsilon/2) + 1))
    q = (1/(math.exp(epsilon/2) + 1))
    for i in range(len(encoded_val)):
        rand = random.random()
        if rand > p:
            #encoded_val[i] = ~encoded_val[i] 
            if encoded_val[i] == 1:
                encoded_val[i] = 0
            else:
                encoded_val[i] = 1  
    return encoded_val


def estimate_rappor(perturbed_values, epsilon, domain):
    d = len(domain)
    p = (math.exp(epsilon/2)/(math.exp(epsilon/2) + 1))
    q = (1/(math.exp(epsilon/2) + 1))
    Iv = np.sum(np.array(perturbed_values), axis=0)
    C_n = [0]*d
    for v in range(d):
        C_n[v]= (Iv[v] - (len(perturbed_values)*q))/(p-q)
    
    C_n = [0 if x<0 else x for x in C_n]
    C_n = [round(x) for x in C_n]
    return C_n


def rappor_experiment(dataset, epsilon, domain):
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        perturbed_list = [perturb_rappor(encode_rappor(val, domain),epsilon,domain) for val in dataset]
        estimated_list = estimate_rappor(perturbed_list, epsilon,domain)
        d = len(domain)
        actual_list = [0]*d  
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1
        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

# OUE

def encode_oue(val, domain):
    return [1 if int(val/RANGE_SIZE)==x else 0 for x in domain]

def perturb_oue(encoded_val, epsilon, domain):
    d = len(domain)
    p_1 = 1/2
    p_0 = (math.exp(epsilon)/(math.exp(epsilon) + 1))
    q = (1/(math.exp(epsilon) + 1))
    for i in range(len(encoded_val)):
        rand = random.random()
        if encoded_val[i] == 1:
            if rand <= p_1:
                encoded_val[i] = 1
            else:
                encoded_val[i] = 0
        else:
            if rand <= q:
                encoded_val[i] = 1
            else:
                encoded_val[i] = 0
    return encoded_val

def estimate_oue(perturbed_values, epsilon, domain):
    d = len(domain)

    Iv = np.sum(np.array(perturbed_values), axis=0)
    C_n = [0]*d
    for v in range(d):
        C_n[v]= (2*(Iv[v]*(math.exp(epsilon) + 1) - len(perturbed_values)))/(math.exp(epsilon)- 1)
    
    C_n = [0 if x<0 else x for x in C_n]
    C_n = [round(x) for x in C_n]
    return C_n

def oue_experiment(dataset, epsilon, domain):
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        perturbed_list = [perturb_oue(encode_oue(val,domain),epsilon,domain) for val in dataset]
        estimated_list = estimate_oue(perturbed_list, epsilon, domain)
        d = len(domain)
        actual_list = [0]*d  
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1
        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

# Binary Local Hashing(BLH)
def compressed_blh(domain):
    g = 2
    array = hash_func_with_modulo(domain,g)
    return array

def perturb_blh(compressed_value,epsilon,domain,val):
    d = len(domain)
    p = math.exp(epsilon) / (math.exp(epsilon) + 1)
    encoded_val = [0]*d
    modulo = int(val/RANGE_SIZE)
    rand = random.random()
    if rand <= p:
        for i in range(len(compressed_value)):
            if modulo in compressed_value[i]:
                for k in compressed_value[i]:
                    encoded_val[k] = 1 
    else:
        for i in range(len(compressed_value)):
            if modulo not in compressed_value[i]:
                for k in compressed_value[i]:
                    encoded_val[k] = 1 
    return encoded_val

def estimate_blh(perturbed_values, epsilon, domain):
    d = len(domain)
    p = math.exp(epsilon) / (math.exp(epsilon)+1)
    Iv = np.sum(np.array(perturbed_values), axis=0)
    C_n = [0]*d
    for v in range(d):
        C_n[v]= (Iv[v] - (len(perturbed_values)*0.5))/(p-0.5)
    C_n = [0 if x<0 else x for x in C_n]
    C_n = [round(x) for x in C_n]
    return C_n

def blh_experiment(dataset, epsilon, domain):
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        perturbed_list = [perturb_blh(compressed_blh(domain),epsilon,domain,val) for val in dataset]
        estimated_list = estimate_blh(perturbed_list, epsilon, domain)
        d = len(domain)
        actual_list = [0]*d
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1
        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT
    
# Optimal Local Hashing OLH 

def compressed_olh(domain,epsilon):
    g = int(math.exp(epsilon) +1)
    array = hash_func_with_modulo(domain,g)
    return array

def perturb_olh(compressed_value,epsilon,domain,val):
    g = int(math.exp(epsilon) +1)
    d = len(domain)
    p = 1/2
    encoded_val = [0]*d
    d = len(domain)
    modulo = (int(val/RANGE_SIZE)) 
    rand = random.random()
    if rand <= p:
        for i in range(len(compressed_value)):
            if modulo in compressed_value[i]:
                for k in compressed_value[i]:
                    encoded_val[k] = 1 
    else:
        index = random.randint(0,g-1)
        while modulo in compressed_value[index]:
            index = random.randint(0,g-1)
        for k in compressed_value[index]:
            encoded_val[k] = 1

    return encoded_val

def estimate_olh(perturbed_values, epsilon, domain):
    d = len(domain)
    g = int(math.exp(epsilon) +1)
    p = (math.exp(epsilon)/(math.exp(epsilon) + g - 1))
    q = (1- p)/(g-1)
    Iv = np.sum(np.array(perturbed_values), axis=0)
    C_n = [0]*d
    for v in range(d):
        C_n[v]= (Iv[v] - (len(perturbed_values)*q))/(p-q)
    C_n = [0 if x<0 else x for x in C_n] # for negative values makes them 0
    C_n = [round(x) for x in C_n]
    return C_n

def olh_experiment(dataset, epsilon, domain):
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        perturbed_list = [perturb_olh(compressed_olh(domain,epsilon),epsilon,domain,val) for val in dataset]
        estimated_list = estimate_olh(perturbed_list, epsilon, domain)
        d = len(domain)
        actual_list = [0]*d
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1

        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

def get_list_form_df(df,month):
    return list(df[month])

# Summation Histogram Encoding SHE
def she_experiment(dataset,epsilon,domain):
    d=len(domain)   
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        client_she = HEClient(epsilon=epsilon, d=d)
        server_she = HEServer(epsilon=epsilon, d=d)
        perturbed_list = [client_she.privatise(int(val/RANGE_SIZE)) for val in dataset]
        server_she.aggregate_all(perturbed_list)
        estimated_list = server_she.estimate_all(range(1,d+1))
        actual_list = [0]*d
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1

        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

# Thresholding Histogram Encoding THE
def the_experiment(dataset,epsilon,domain):
    d=len(domain)
   
    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        client_she = HEClient(epsilon=epsilon, d=d)
        server_she = HEServer(epsilon=epsilon, d=d, use_the=True)
        perturbed_list = [client_she.privatise(int(val/RANGE_SIZE)) for val in dataset]
        server_she.aggregate_all(perturbed_list)
        estimated_list = server_she.estimate_all(range(1,d+1))
        actual_list = [0]*d
        for i in dataset:
            actual_list[int(i/RANGE_SIZE)] += 1

        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT

def combined_protocol_experiment(dataset,epsilon, domain):
    weight = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    #GRR RAPPOR OUE BLh OLH SHE THE 
    if(epsilon<1.0):
        weight = [0.0,0.4,0.3,0.3,0.0,0.0,0,0]
    elif(epsilon <= 3.0):
        weight = [0.1,0.3,0.2,0.2,0.0,0.1,0,1]
    else:
        weight = [0.45,0.1,0.45,0.0,0.0,0.0,0.0]
    
    d = len(domain)
    actual_list = [0]*d  
    for i in dataset:
        actual_list[int(i/RANGE_SIZE)] += 1

    average_aggregate_error = 0
    for i in range(EXPERIMENT_COUNT):
        #GRR part
        grr_perturbed_list = [perturb_grr(val, epsilon, domain) for val in dataset]
        grr_estimated_list = estimate_grr(grr_perturbed_list, epsilon, domain)

        #RAPPOR part
        rappor_perturbed_list = [perturb_rappor(encode_rappor(val, domain),epsilon,domain) for val in dataset]
        rappor_estimated_list = estimate_rappor(rappor_perturbed_list, epsilon,domain)

        #OUE part
        oue_perturbed_list = [perturb_oue(encode_oue(val,domain),epsilon,domain) for val in dataset]
        oue_estimated_list = estimate_oue(oue_perturbed_list, epsilon, domain)

        #BLH Part   
        blh_perturbed_list = [perturb_blh(compressed_blh(domain),epsilon,domain,val) for val in dataset]
        blh_estimated_list = estimate_blh(blh_perturbed_list, epsilon, domain)
    
        #OLH Part
        olh_perturbed_list = [perturb_olh(compressed_olh(domain,epsilon),epsilon,domain,val) for val in dataset]
        olh_estimated_list = estimate_olh(olh_perturbed_list, epsilon, domain)

        #SHE Part
        client_she = HEClient(epsilon=epsilon, d=d)
        server_she = HEServer(epsilon=epsilon, d=d)
        she_perturbed_list = [client_she.privatise(int(val/RANGE_SIZE)) for val in dataset]
        server_she.aggregate_all(she_perturbed_list)
        she_estimated_list = server_she.estimate_all(range(1,d+1))

        #THE Part
        client_the = HEClient(epsilon=epsilon, d=d)
        server_the = HEServer(epsilon=epsilon, d=d, use_the=True)
        the_perturbed_list = [client_the.privatise(int(val/RANGE_SIZE)) for val in dataset]
        server_the.aggregate_all(the_perturbed_list)
        the_estimated_list = server_the.estimate_all(range(1,d+1))

        total_list=np.array([grr_estimated_list,rappor_estimated_list,oue_estimated_list,blh_estimated_list,olh_estimated_list,she_estimated_list,the_estimated_list])

        weight = np.array(weight)
        estimated_list = np.sum([weight[i]*total_list[i] for i in range(7)], axis=0)
    
        average_aggregate_error += calculate_average_error(actual_list, estimated_list)
    return average_aggregate_error/EXPERIMENT_COUNT


def main():
    epsilon_array = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = read_dataset("CC_LCL-FullData_formatted_final.csv")
    print("DISTRIBUTION ESTIMATION OVER MONTHS")
    print("-" * 50)
    for month in df.columns[7:]:
        error_list = []
        grr_error_list = []        
        rappor_error_list = []
        oue_error_list = []
        blh_error_list = []
        olh_error_list = []
        she_error_list = []
        the_error_list = []
        combined_error_list = []
        
        time_list = []
        grr_time_list = 0.0       
        rappor_time_list = 0.0
        oue_time_list = 0.0
        blh_time_list = 0.0
        olh_time_list = 0.0
        she_time_list = 0.0
        the_time_list = 0.0
        combined_time_list = 0.0

        print("Date: " + month + "\n")
        dataset = list(df[month]) 
        domain = get_domain(dataset)

        print("GRR EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = grr_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            grr_error_list.append(error)
            end= time.time()
            grr_time_list += (end - start)
        error_list.append(grr_error_list)
        time_list.append(grr_time_list/len(epsilon_array))
        print("*" * 50)

        print("RAPPOR EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = rappor_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            rappor_error_list.append(error)
            end= time.time()
            rappor_time_list += (end - start)
        time_list.append(rappor_time_list/len(epsilon_array))
        error_list.append(rappor_error_list)
        print("*" * 50)
        
        print("OUE EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = oue_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            oue_error_list.append(error)
            end= time.time()
            oue_time_list += (end - start)
        time_list.append(oue_time_list/len(epsilon_array))
        error_list.append(oue_error_list)
        print("*" * 50)

        print("BLH EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = blh_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            blh_error_list.append(error)
            end= time.time()
            blh_time_list += (end - start)
        time_list.append(blh_time_list/len(epsilon_array))
        error_list.append(blh_error_list)
        print("*" * 50)

        print("OLH EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = olh_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            olh_error_list.append(error)
            end= time.time()
            olh_time_list += (end - start)
        time_list.append(olh_time_list/len(epsilon_array))
        error_list.append(olh_error_list)
        print("*" * 50)

        print("SHE EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = she_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))
            she_error_list.append(error)
            end= time.time()
            she_time_list += (end - start)
        time_list.append(she_time_list/len(epsilon_array))
        error_list.append(she_error_list)
        print("*" * 50)

        print("THE EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = the_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))  
            the_error_list.append(error)
            end= time.time()
            the_time_list += (end - start)
        time_list.append(the_time_list/len(epsilon_array))
        error_list.append(the_error_list)       
        print("*" * 50)
        
        print("COMBINED PROTOCOL EXPERIMENT")
        for epsilon in epsilon_array:
            start = time.time()
            error = combined_protocol_experiment(dataset, epsilon, domain)
            print("e={}, Error: {}".format(epsilon, error))  
            combined_error_list.append(error)
            end= time.time()
            combined_time_list += (end - start)
        time_list.append(combined_time_list/len(epsilon_array))
        error_list.append(combined_error_list)
        
        
        print("Run time of the algorithms respectively: ",time_list)

        error_list_year.append(error_list)
        
if __name__ == "__main__":
    main()
