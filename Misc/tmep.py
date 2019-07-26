#import matplotlib.pyplot as plt

provs = ['Sask','Alberta', 'Man']
print(provs)
pops = [5, 7, 9]
print(pops)
prov_pops = []
#big_list = [provs,pops]
#print(big_list)


for i in range(len(provs)):
    prov_pops.append(provs[i])
    prov_pops.append(pops[i])
print(prov_pops)

prov_props = []
for i in range(len(prov_pops)-1):
    big_list = [prov_pops[i],prov_pops[i+1]]
    prov_props.append(big_list)
    i=i+1

print(prov_props)

msgs = [['hello',4],['goodbye',8]]


#Ch 9
#Set variable to passcode
#set another variable number of attempts equal to 0
#make a while loop
    #while (passcode = string input password)

pswd = 'cats'
nAtmp = 1
#input_pass = str(input('Please Enter Password'))
#while pswd != input_pass:
#    nAtmp = nAtmp + 1
#    input_pass = str(input('Please Enter Password'))
#print(nAtmp)


new_list=[]
for i in range(300,600):
    if (i%3==0) and (i%5==0):
#        new_list.append(i)
#print("ch9:",len(new_list))

def report_card(grade):
    print(grade)



report_card(80)