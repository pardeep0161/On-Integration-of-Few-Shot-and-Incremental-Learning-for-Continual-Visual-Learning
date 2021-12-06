import fewShotIL
import fewShotLearn
import iCARLIL

default_path = '/home/cis/ppathania/Proto-Networking-iCarl-Integration-main/previous models findings/'
print('####################################################################################################################################')
folder= default_path+'run4'
print('-------------iCarl_il-------------')
iCARLIL.iCarl_il(folder, full=False)
print('-------------few_shot-------------')
fewShotLearn.few_shot(folder, full=False)
print('-------------few_shot_il----------')
fewShotIL.few_shot_il(folder, full=False)
print('----------------------------------')

print('####################################################################################################################################')

folder= default_path+'run5'
print('-------------iCarl_il-------------')
iCARLIL.iCarl_il(folder, full=True)
print('-------------few_shot-------------')
fewShotLearn.few_shot(folder, full=True)
print('-------------few_shot_il----------')
fewShotIL.few_shot_il(folder, full=True)
print('----------------------------------')


print('####################################################################################################################################')

folder= default_path+'run6'
print('-------------iCarl_il-------------')
iCARLIL.iCarl_il(folder, full=True, max_models=True)
print('-------------few_shot_il----------')
fewShotIL.few_shot_il(folder, full=True, max_models=True)
print('----------------------------------')