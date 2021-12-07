import json
from easydict import EasyDict
import os
import datetime
import torch

# self.weight_degrees = [2,3,4,5]
# self.subgraph_degrees = [1,2,3,4,5,6]
# self.degree_mapping = {'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'5'} # A mapping from sub graph degrees to the associated weight matrix given by the weight degree.
WEIGHT_DEGREES = {'COLLAB':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59,61,63,65,69,77,89,90,92,98,99,238,239,240,241,242,243,245,249,255,276,296,297], 
#                   'IMDBBINARY':[2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,26,29,32], 
                  'IMDBBINARY':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,49,50,51,54,55,56,57,58,59,60,62,63,64,65,68,71,83,86,135],
                  'IMDBMULTI':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,29,30,31,32,37,38], 
                  'MUTAG':[2,3,4], 
                  'NCI1':[2,3,4,5], 
                  'NCI109':[2,3,4,5], 
#                   'PROTEINS':[2,3,4,5,6,7,8,9], 
                  'PROTEINS':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,40], 
                  'PTC':[2,3,4,5]}
SUBGRAPH_DEGREES = {'COLLAB':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,201,202,204,207,208,210,211,212,213,214,215,216,221,222,224,227,228,229,235,236,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,334,335,336,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,354,355,357,358,359,362,363,364,365,366,367,369,370,371,372,374,375,380,381,385,387,394,400,403,409,415,418,422,423,424,425,427,440,443,444,451,482,486,491], 
                    'IMDBBINARY':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,49,50,51,54,55,56,57,58,59,60,62,63,64,65,68,71,83,86,135], 
                    'IMDBMULTI':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,49,50,51,53,56,58,59,62,67,76,77,88], 
                    'MUTAG':[2,3,4,5], 
                    'NCI1':[1,2,3,4,5], 
                    'NCI109':[1,2,3,4,5,6], 
#                     'PROTEINS':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,25,26], 
                    'PROTEINS':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,38,39,40,41,42,43,45,46,50,54,57], 
                    'PTC':[2,3,4,5]}
DEGREE_MAPPING = {#'COLLAB':{'1':,'2':,'3':,'4':,'5':,'6':,'7':,'8':,'9':,'10':,'11':,'12':,'13':,'14':,'15':,'16':,'17':,'18':,'19':,'20':,'21':,'22':,'23':,'24':,'25':,'26':,'27':,'28':,'29':,'30':,'31':,'32':,'33':,'34':,'35':,'36':,'37':,'38':,'39':,'40':,'41':,'42':,'43':,'44':,'45':,'46':,'47':,'48':,'49':,'50':,'51':,'52':,'53':,'54':,'55':,'56':,'57':,'58':,'59':,'60':,'61':,'62':,'63':,'64':,'65':,'66':,'67':,'68':,'69':,'71':,'72':,'73':,'74':,'75':,'76':,'77':,'78':,'79':,'80':,'81':,'82':,'83':,'84':,'85':,'86':,'87':,'88':,'89':,'90':,'91':,'92':,'93':,'94':,'95':,'96':,'97':,'98':,'99':,'100':,'101':,'102':,'103':,'104':,'105':,'106':,'107':,'108':,'109':,'110':,'111':,'112':,'113':,'114':,'115':,'116':,'117':,'118':,'119':,'120':,'121':,'122':,'123':,'124':,'125':,'126':,'127':,'128':,'129':,'130':,'131':,'132':,'133':,'134':,'135':,'136':,'137':,'138':,'139':,'140':,'141':,'142':,'143':,'144':,'145':,'146':,'147':,'148':,'149':,'150':,'151':,'152':,'153':,'154':,'155':,'156':,'157':,'158':,'159':,'160':,'161':,'162':,'163':,'164':,'165':,'166':,'167':,'168':,'169':,'170':,'172':,'173':,'174':,'175':,'176':,'177':,'178':,'179':,'180':,'181':,'182':,'183':,'184':,'185':,'186':,'187':,'188':,'189':,'190':,'191':,'192':,'194':,'195':,'196':,'201':,'202':,'204':,'207':,'208':,'210':,'211':,'212':,'213':,'214':,'215':,'216':,'221':,'222':,'224':,'227':,'228':,'229':,'235':,'236':,'238':,'239':,'240':,'241':,'242':,'243':,'244':,'245':,'246':,'247':,'248':,'249':,'250':,'251':,'252':,'253':,'254':,'255':,'256':,'257':,'258':,'259':,'260':,'261':,'262':,'263':,'264':,'265':,'266':,'267':,'268':,'269':,'270':,'271':,'272':,'273':,'274':,'275':,'276','277','278','279','280','281','282','283','284','285','286','287','288','289','290','291','292','293','294','295','296','297','298','299','300','301','302','303','304','305','306','307','308','309','310','311','312','313','314','315','316','317','318','319','320','321','322','323','324','325','326','327','328','329','330','331','332','334','335','336','338','339','340','341','342','343','344','345','346','347','348','349','350','351','352','354','355','357','358','359','362','363','364','365','366','367','369','370','371','372','374','375','380','381','385','387','394','400','403','409','415','418','422','423','424','425','427','440','443','444','451','482','486','491'}, 
#                   'IMDBBINARY':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12':'12', '13':'12', '14':'14', '15':'14', '16':'16', '17':'16', '18':'18', '19':'18', '20':'20', '21':'20', '22':'20', '23':'20', '24':'20', '25':'20', '26':'26', '27':'26', '28':'26', '29':'26', '30':'26', '31':'26', '32':'32', '33':'32', '34':'32', '35':'32', '36':'32', '37':'32', '38':'32', '39':'32', '40':'32', '41':'32', '42':'32', '43':'32', '44':'32', '45':'32', '48':'32', '49':'32', '50':'32', '51':'32', '54':'32', '55':'32', '56':'32', '57':'32', '58':'32', '59':'32', '60':'32', '62':'32', '63':'32', '64':'32', '65':'32', '68':'32', '71':'32', '83':'32', '86':'32', '135':'32'}, 
                  'IMDBBINARY':{'1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16', '17':'17', '18':'18', '19':'19', '20':'20', '21':'21', '22':'22', '23':'23', '24':'24', '25':'25', '26':'26', '27':'27', '28':'28', '29':'29', '30':'30', '31':'31', '32':'32', '33':'33', '34':'34', '35':'35', '36':'36', '37':'37', '38':'38', '39':'39', '40':'40', '41':'41', '42':'42', '43':'43', '44':'44', '45':'45', '48':'48', '49':'49', '50':'50', '51':'51', '54':'54', '55':'55', '56':'56', '57':'57', '58':'58', '59':'59', '60':'60', '62':'62', '63':'63', '64':'64', '65':'65', '68':'68', '71':'71', '83':'83', '86':'86', '135':'135'},
                  'IMDBMULTI':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16', '17':'17', '18':'18', '19':'19', '20':'20', '21':'20', '22':'22', '23':'22', '24':'22', '25':'22', '26':'22', '27':'22', '28':'22', '29':'29', '30':'30', '31':'31', '32':'32', '33':'32', '34':'32', '35':'32', '36':'32', '37':'37', '38':'38', '39':'38', '40':'38', '42':'38', '43':'38', '44':'38', '45':'38', '46':'38', '47':'38', '48':'38', '49':'38', '50':'38', '51':'38', '53':'38', '56':'38', '58':'38', '59':'38', '62':'38', '67':'38', '76':'38', '77':'38', '88':'38'}, 
                  'MUTAG':{'2':'2', '3':'3', '4':'4', '5':'4'}, 
                  'NCI1':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5'}, 
                  'NCI109':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'5'}, 
#                   'PROTEINS':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'9', '11':'9', '12':'9', '13':'9', '14':'9', '16':'9', '25':'9', '26':'9'}, 
                  'PROTEINS':{'1':'2', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16', '17':'17', '18':'18', '19':'19', '20':'20', '21':'21', '22':'22', '23':'22', '24':'22', '25':'22', '26':'22', '27':'22', '28':'22', '29':'22', '30':'22', '32':'22', '33':'22', '34':'22', '35':'22', '38':'22', '39':'22', '40':'40', '41':'40', '42':'40', '43':'40', '45':'40', '46':'40', '50':'40', '54':'40', '57':'40'}, 
                  'PTC':{'2':'2', '3':'3', '4':'4', '5':'5'}}

NUM_LABELS = {'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'DD':89}
NUM_CLASSES = {'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2}
# LEARNING_RATES = {'COLLAB': 0.00256, 'IMDBBINARY': 0.00064, 'IMDBMULTI': 0.00064, 'MUTAG': 0.001, 'NCI1':0.0006, 'NCI109':0.00256, 'PROTEINS': 0.00064, 'PTC': 0.0008325514166712444}
# LEARNING_RATES = {'COLLAB': 0.00256, 'IMDBBINARY': 0.00064, 'IMDBMULTI': 0.00064, 'MUTAG': 0.0015, 'NCI1':0.0006, 'NCI109':0.0006, 'PROTEINS': 0.00064, 'PTC': 0.0008325514166712444}
LEARNING_RATES = {'COLLAB': 0.00256, 'IMDBBINARY': 0.00064, 'IMDBMULTI': 0.00064, 'MUTAG': 0.0015, 'NCI1':0.0006, 'NCI109':0.0006, 'PROTEINS': 0.00064, 'PTC': 0.0006}
DECAY_RATES = {'COLLAB': 0.7, 'IMDBBINARY': 0.4, 'IMDBMULTI': 0.7, 'MUTAG': 0.7, 'NCI1':0.7, 'NCI109':0.7, 'PROTEINS': 0.7, 'PTC': 0.7}
# CHOSEN_EPOCH = {'COLLAB': 100, 'IMDBBINARY': 40, 'IMDBMULTI': 150, 'MUTAG': 200, 'NCI1': 99, 'NCI109': 99, 'PROTEINS': 20, 'PTC': 36}
CHOSEN_EPOCH = {'COLLAB': 50, 'IMDBBINARY': 50, 'IMDBMULTI': 50, 'MUTAG': 50, 'NCI1': 50, 'NCI109': 50, 'PROTEINS': 50, 'PTC': 50}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.num_classes = NUM_CLASSES[config.dataset_name]
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join(os.path.dirname(__file__),"../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join(os.path.dirname(__file__),"../experiments", config.parent_dir, "checkpoint/")
    config.weight_degrees = WEIGHT_DEGREES[config.dataset_name]
    config.subgraph_degrees = SUBGRAPH_DEGREES[config.dataset_name]
    config.degree_mapping = DEGREE_MAPPING[config.dataset_name]
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.learning_rate = LEARNING_RATES[config.dataset_name]
        config.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    if torch.cuda.is_available():
        config.cuda = True
        config.device = 'cuda'
    else:
        config.cuda = False
        config.device = 'cpu'
    return config

if __name__ == '__main__':
    config = process_config('../configs/example.json')
    print(config.values())