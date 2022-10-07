import h5py, os
import nibabel as nib

train_split = {0: ['Brats18_2013_23_1', 'Brats18_CBICA_AAL_1', 'Brats18_CBICA_AOO_1', 'Brats18_CBICA_APZ_1', 'Brats18_CBICA_AQT_1', 'Brats18_CBICA_AQV_1', 'Brats18_CBICA_ARF_1', 'Brats18_CBICA_ASE_1', 'Brats18_CBICA_AWG_1', 'Brats18_CBICA_AXW_1', 'Brats18_CBICA_BHB_1', 'Brats18_TCIA01_411_1', 'Brats18_TCIA02_208_1', 'Brats18_TCIA02_608_1', 'Brats18_TCIA03_338_1', 'Brats18_TCIA05_478_1', 'Brats18_TCIA06_372_1']
,1: ['Brats18_2013_19_1', 'Brats18_2013_27_1', 'Brats18_CBICA_ABN_1', 'Brats18_CBICA_ABO_1', 'Brats18_CBICA_ALU_1', 'Brats18_CBICA_AMH_1', 'Brats18_CBICA_AQJ_1', 'Brats18_CBICA_AQR_1', 'Brats18_CBICA_AQU_1', 'Brats18_TCIA01_401_1', 'Brats18_TCIA02_321_1', 'Brats18_TCIA02_471_1', 'Brats18_TCIA02_473_1', 'Brats18_TCIA04_192_1', 'Brats18_TCIA04_328_1', 'Brats18_TCIA08_162_1', 'Brats18_TCIA08_234_1']
,2: ['Brats18_2013_17_1', 'Brats18_2013_2_1', 'Brats18_CBICA_ABY_1', 'Brats18_CBICA_AQG_1', 'Brats18_CBICA_AQP_1', 'Brats18_CBICA_ASY_1', 'Brats18_CBICA_AVG_1', 'Brats18_CBICA_AXJ_1', 'Brats18_CBICA_AYA_1', 'Brats18_TCIA02_290_1', 'Brats18_TCIA02_300_1', 'Brats18_TCIA02_331_1', 'Brats18_TCIA02_377_1', 'Brats18_TCIA03_138_1', 'Brats18_TCIA03_375_1', 'Brats18_TCIA04_149_1', 'Brats18_TCIA04_437_1']
,3: ['Brats18_CBICA_ALX_1', 'Brats18_CBICA_ANG_1', 'Brats18_CBICA_AQZ_1', 'Brats18_CBICA_ATV_1', 'Brats18_CBICA_AUQ_1', 'Brats18_CBICA_AWI_1', 'Brats18_TCIA01_147_1', 'Brats18_TCIA01_186_1', 'Brats18_TCIA02_491_1', 'Brats18_TCIA02_606_1', 'Brats18_TCIA03_121_1', 'Brats18_TCIA03_257_1', 'Brats18_TCIA04_479_1', 'Brats18_TCIA06_165_1', 'Brats18_TCIA08_205_1', 'Brats18_TCIA08_218_1', 'Brats18_TCIA08_280_1']
,4: ['Brats18_2013_18_1', 'Brats18_2013_20_1', 'Brats18_CBICA_ABE_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AQN_1', 'Brats18_CBICA_ARW_1', 'Brats18_CBICA_ATB_1', 'Brats18_CBICA_ATD_1', 'Brats18_CBICA_AYW_1', 'Brats18_TCIA01_203_1', 'Brats18_TCIA02_135_1', 'Brats18_TCIA03_474_1', 'Brats18_TCIA06_184_1', 'Brats18_TCIA06_247_1', 'Brats18_TCIA06_332_1', 'Brats18_TCIA08_436_1', 'Brats18_TCIA08_469_1']
,5: ['Brats18_CBICA_ABM_1', 'Brats18_CBICA_AOH_1', 'Brats18_CBICA_AOP_1', 'Brats18_CBICA_ARZ_1', 'Brats18_CBICA_ASU_1', 'Brats18_CBICA_AXN_1', 'Brats18_TCIA01_150_1', 'Brats18_TCIA01_378_1', 'Brats18_TCIA02_168_1', 'Brats18_TCIA02_171_1', 'Brats18_TCIA02_198_1', 'Brats18_TCIA02_226_1', 'Brats18_TCIA02_309_1', 'Brats18_TCIA02_322_1', 'Brats18_TCIA03_199_1', 'Brats18_TCIA03_498_1', 'Brats18_TCIA08_167_1']
,6: ['Brats18_2013_5_1', 'Brats18_CBICA_AQD_1', 'Brats18_CBICA_AQQ_1', 'Brats18_CBICA_ASK_1', 'Brats18_CBICA_ASW_1', 'Brats18_CBICA_AUN_1', 'Brats18_CBICA_AWH_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA01_235_1', 'Brats18_TCIA02_222_1', 'Brats18_TCIA02_314_1', 'Brats18_TCIA02_394_1', 'Brats18_TCIA06_603_1', 'Brats18_TCIA08_105_1', 'Brats18_TCIA08_242_1', 'Brats18_TCIA08_278_1', 'Brats18_TCIA08_406_1']
,7: ['Brats18_2013_22_1', 'Brats18_2013_26_1', 'Brats18_CBICA_ANP_1', 'Brats18_CBICA_AOD_1', 'Brats18_CBICA_AOZ_1', 'Brats18_CBICA_APR_1', 'Brats18_CBICA_ASA_1', 'Brats18_CBICA_ASG_1', 'Brats18_CBICA_ASO_1', 'Brats18_CBICA_AXL_1', 'Brats18_TCIA01_425_1', 'Brats18_TCIA01_460_1', 'Brats18_TCIA02_179_1', 'Brats18_TCIA03_419_1', 'Brats18_TCIA04_343_1', 'Brats18_TCIA05_396_1', 'Brats18_TCIA05_444_1']
,8: ['Brats18_2013_11_1', 'Brats18_2013_13_1', 'Brats18_CBICA_AAB_1', 'Brats18_CBICA_ASH_1', 'Brats18_CBICA_ATF_1', 'Brats18_CBICA_ATX_1', 'Brats18_CBICA_AXO_1', 'Brats18_CBICA_BHM_1', 'Brats18_TCIA01_201_1', 'Brats18_TCIA01_221_1', 'Brats18_TCIA01_412_1', 'Brats18_TCIA01_499_1', 'Brats18_TCIA02_368_1', 'Brats18_TCIA03_265_1', 'Brats18_TCIA03_296_1', 'Brats18_TCIA04_361_1', 'Brats18_TCIA08_319_1']
,9: ['Brats18_2013_12_1', 'Brats18_2013_4_1', 'Brats18_CBICA_AAP_1', 'Brats18_CBICA_ALN_1', 'Brats18_CBICA_ANZ_1', 'Brats18_CBICA_ATP_1', 'Brats18_CBICA_AUR_1', 'Brats18_TCIA01_131_1', 'Brats18_TCIA01_231_1', 'Brats18_TCIA01_335_1', 'Brats18_TCIA02_117_1', 'Brats18_TCIA02_118_1', 'Brats18_TCIA02_607_1', 'Brats18_TCIA06_409_1', 'Brats18_TCIA08_113_1']}

test_split = {0:['Brats18_2013_10_1', 'Brats18_2013_14_1', 'Brats18_2013_21_1', 'Brats18_2013_25_1', 'Brats18_2013_3_1', 'Brats18_2013_7_1', 'Brats18_CBICA_AAG_1', 'Brats18_CBICA_ABB_1', 'Brats18_CBICA_AME_1', 'Brats18_CBICA_APY_1', 'Brats18_CBICA_AQA_1', 'Brats18_CBICA_AQO_1', 'Brats18_CBICA_AQY_1', 'Brats18_CBICA_ASN_1', 'Brats18_CBICA_ASV_1', 'Brats18_CBICA_AVJ_1', 'Brats18_CBICA_AVV_1', 'Brats18_CBICA_AXM_1', 'Brats18_CBICA_AXQ_1', 'Brats18_CBICA_AYI_1', 'Brats18_CBICA_AYU_1', 'Brats18_CBICA_AZH_1', 'Brats18_CBICA_BFB_1', 'Brats18_CBICA_BFP_1', 'Brats18_CBICA_BHK_1', 'Brats18_TCIA01_180_1', 'Brats18_TCIA01_190_1', 'Brats18_TCIA01_390_1', 'Brats18_TCIA01_429_1', 'Brats18_TCIA01_448_1', 'Brats18_TCIA02_151_1', 'Brats18_TCIA02_274_1', 'Brats18_TCIA02_283_1', 'Brats18_TCIA02_370_1', 'Brats18_TCIA02_374_1', 'Brats18_TCIA02_430_1', 'Brats18_TCIA02_455_1', 'Brats18_TCIA02_605_1', 'Brats18_TCIA03_133_1', 'Brats18_TCIA04_111_1', 'Brats18_TCIA05_277_1', 'Brats18_TCIA06_211_1']}

def split_data(split_type="train"):
    path = "../../datasets/brats18/HGG"
    dst = f"../../datasets/BRATS/tumor_split_{split_type}"
    if not os.path.exists(dst):
        os.mkdir(dst)
    if split_type == 'train':
        print("==== train split ====")
        for key in train_split:
            new_path = os.path.join(dst, str(key))
            for each_name in train_split[key]:
                this_path = os.path.join(path, each_name)
                print(f"{this_path} -> {new_path}")
                os.system(f"cp -r {this_path} {new_path}")

        for key in train_split:
            files = os.listdir(os.path.join(dst, str(key)))
            for f in files:
                file_path = os.path.join(dst, str(key), f)
                if not os.path.isdir(file_path):
                    print(f"remove {file_path}")
                    os.system(f"rm {file_path}")
    elif split_type == 'test':
        print("==== test split ====")
        for key in test_split:
            new_path = os.path.join(dst, str(key))
            for each_name in train_split[key]:
                this_path = os.path.join(path, each_name)
                print(f"{this_path} -> {new_path}")
                os.system(f"cp -r {this_path} {new_path}")

        for key in test_split:
            files = os.listdir(os.path.join(dst, str(key)))
            for f in files:
                file_path = os.path.join(dst, str(key), f)
                if not os.path.isdir(file_path):
                    print(f"remove {file_path}")
    print("finished!")

def build_data(data_type="train"):
    path = "../../datasets/BRATS/tumor_split"
    assert os.path.exists(path)
    dst = "../../datasets/BRATS/splitedH5"
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    dirs = os.listdir(path)
    idx = 0
    print(path)
    for dir in dirs:
        subs = os.listdir(os.path.join(path, dir))
        
        file_name = f"BraTS18_tumor_size_{dir}.h5"
        print(f">>> building {file_name}")
        result_file = h5py.File(os.path.join(dst, file_name), 'w')

        for sub_dir in subs:
            prefix_path = os.path.join(path, dir)
            flair = nib.load(os.path.join(prefix_path, sub_dir, sub_dir+"_flair.nii.gz")).get_fdata()
            seg = nib.load(os.path.join(prefix_path, sub_dir, sub_dir+"_seg.nii.gz")).get_fdata()
            t1 = nib.load(os.path.join(prefix_path, sub_dir, sub_dir+"_t1.nii.gz")).get_fdata()
            t1ce = nib.load(os.path.join(prefix_path, sub_dir, sub_dir+"_t1ce.nii.gz")).get_fdata()
            t2 = nib.load(os.path.join(prefix_path, sub_dir, sub_dir+"_t2.nii.gz")).get_fdata()

            result_file.create_dataset(f"{data_type}/{idx}/flair", data=flair)
            db = result_file.create_dataset(f"{data_type}/{idx}/seg", data=seg)
            result_file.create_dataset(f"{data_type}/{idx}/t1", data=t1)
            result_file.create_dataset(f"{data_type}/{idx}/t1ce", data=t1ce)
            result_file.create_dataset(f"{data_type}/{idx}/t2", data=t2)

            db.attrs['id'] = sub_dir
            print(f"***Finish create one database:{prefix_path}:{sub_dir},***")
            idx += 1
    result_file.close()
    print("Finished!")

if __name__ == "__main__":
    build_data()


