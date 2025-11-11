#!bin/sh

## global variable for usps experiment
device=0

gv_v=1
gv_lrtheta=1e-1
gv_nthetaupdate=101
gv_lrtheta=1e-1


# nmem_list="50 100 250 500 1000" 
# seed_list="111 112 113 114 115"

nmem_list="50"  ## for test
seed_list="111"




gv_delta=1e-1
em_noise=1e-4
gv_nmemupdate=10 

for nmem in $nmem_list
do
    gv_meminit=random
    for seed in $seed_list
    do
        ftype=cliprn50
        CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_ourem_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
                                                                                   --nmem $nmem --nmemupdate $gv_nmemupdate --emnoise $em_noise --v $gv_v --seed $seed;


        CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_basereplay_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
                                                                                   --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;


        CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_baselambda_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
                                                                                   --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;


    done

done




# for nmem in $nmem_list
# do
#     gv_meminit=random
#     for seed in $seed_list
#     do

#         ftype=clipvitb32
#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_ourem_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                    --nmem $nmem --nmemupdate $gv_nmemupdate --emnoise $em_noise --v $gv_v --seed $seed;

#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_basereplay_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                     --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;


#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_baselambda_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                    --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;

#     done

# done



# for nmem in $nmem_list
# do
#     gv_meminit=random
#     for seed in $seed_list
#     do

#         ftype=clipvitl114
#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_ourem_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                    --nmem $nmem --nmemupdate $gv_nmemupdate --emnoise $em_noise --v $gv_v --seed $seed;

#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_basereplay_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                     --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;


#         CUDA_VISIBLE_DEVICES=$device python3 main_splitcifar100_baselambda_batch.py --ftype $ftype --lrtheta $gv_lrtheta --nthetaupdate $gv_nthetaupdate --delta $gv_delta --meminit $gv_meminit \
#                                                                                    --nmem $nmem --nmemupdate $gv_nmemupdate --v $gv_v --seed $seed;

#     done

# done


