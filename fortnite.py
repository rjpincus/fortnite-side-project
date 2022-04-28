# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:19:59 2021

@author: Richard Pincus

Side project 2

Fortnite Weapon Classification 

Dec192021
"""

#libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
# from PIL import Image
# import cv2
import boto3
import csv
from pandasql import sqldf  ##Based on SQLite
from sklearn.impute import KNNImputer
# from sklearn.model_selection import train_test_split
# from skmultilearn.model_selection import iterative_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
import random
from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import SMOTENC



###############
##
## Web scraping version
##
###############





##
##Read in HTML code from website
##

# url = urllib.request.urlopen( 'https://fortnite-archive.fandom.com/wiki/All_Weapons' )
# doc = url.read()
# tree = BeautifulSoup( doc )




##
#Save as txt file to then read in to avoid any website updates
##

# url_get = requests.get('https://fortnite-archive.fandom.com/wiki/All_Weapons')
# soup = BeautifulSoup(url_get.content, 'lxml')

# with open("C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/fortnite_html_14Jan2022.txt", 'w', encoding='utf-8') as f_out:
#     f_out.write(soup.prettify())


##
##Read in txt file to get HTML code from day it was pulled
##

hmtl_txt = open("C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/fortnite_html_14Jan2022.txt", encoding='utf-8')
doc = hmtl_txt.read()
tree = BeautifulSoup(doc,"html.parser")



#number of data points        
nod = 194


#Get data
i = -1
rare=[]
wclass=[]
name=[]
bt=[]
dps=[]
damage=[]
fire=[]
mag=[]
reload=[]
sd=[]

#Scrape data
# #Weapon classes
# links0 = tree.find_all( 'h2' )#[0:50]
# for l in links0:
#     if l.span != None:
#         wclass.append(l.span.get( 'id' ))
        
#All other weapon attributes
links = tree.find_all( ['h2','table'] )#[0:5]
class_counter = -1
for l in links:
    # print(l)
    if l.span != None:
         current_class = l.span.get( 'id' )
         if current_class == None:
             if l.tr.th.a.get( 'title' ) == 'Rocket Launcher (Battle Royale)':
                 current_class = 'Explosives'
        
    
    #Ensure the Weapon 'Details' are on the website
    if l.get( 'class' ) != None and l.get( 'class' )[0] == 'miniinfoboxtable':
        
        
                    
        # if l.get( 'class' )[1][0:7] == 'infobox':
        if len(l.find_all( 'tr' )) >= 5:
            i += 1

            #Add Class
            wclass.append(current_class)
        
            #Rarity
            if l.get( 'class' )[1][7:] == 'Icon':
                rare.append((l.get( 'class' )[1][7:] + ' ' + l.get( 'class' )[2]))
            else:
                rare.append(l.get( 'class' )[1][7:])
        
            #Get each attribute for the weapons
            subtables = l.find_all( 'tr' )
            for s in subtables:
                    
                #Name
                if s.th != None:
                    if s.th.a != None:
                        tag = BeautifulSoup(str(s.th.a), 'html.parser').a
                        tag1 = tag.contents
                        print(tag1[0].strip())
                        name.append(tag1[0].strip())
                    elif s.th.a == None and s.th.get( 'class' )[0] == 'infoboxname':
                        print(s.th.contents[0].strip())
                        name.append(s.th.contents[0].strip())
                        
                    
                    
                        
                    #Bullet Type with DPS and Damage for 'Lever Action Rifle' imputed as 9999 for Missing placeholder
                    elif s.th.contents[0].strip() == 'Bullet Type':
                        bt.append(s.td.a.get( 'title'))
                        if name[i] == 'Lever Action Rifle':
                            # print('Here')
                            dps.append(9999.0)
                            damage.append(9999)
                     
                    #DPS
                    elif s.th.contents[0].strip() == 'DPS':
                        dps.append(float(s.td.contents[0].strip()))
                        
                    #Damage
                    elif s.th.contents[0].strip() == 'Damage':
                        damage.append(int(s.td.contents[0].strip()))
                        
                    #Fire Rate
                    elif s.th.contents[0].strip() == 'Fire Rate':
                        fire.append(float(s.td.contents[0].strip()))
                        
                    #Magazine Size
                    elif s.th.contents[0].strip() == 'Magazine Size':
                        try:
                            mag.append(int(s.td.contents[0].strip()))
                        except:
                            if s.td.contents[0].strip() == '∞':
                                mag.append(1000)
                        
                    #Reload Time and Structure Damage for "Kit's Shockwave Launcher"
                    elif s.th.contents[0].strip() == 'Reload Time':
                        reload.append(float(s.td.contents[0].strip()))
                        if name[i] == "Kit's Shockwave Launcher":
                            sd.append(9999)
                        
                    #Structure Damage
                    elif s.th.contents[0].strip() == 'Structure Damage':
                        sd.append(int(s.td.contents[0].strip()))
                        
                    
#Clean up Weapon Class names
for i in range(len(wclass)):
    if wclass[i] == 'Submachine_Guns':
        wclass[i] = 'Submachine Gun'
    elif wclass[i] == 'Shotguns':
        wclass[i] = 'Shotgun'
    elif wclass[i] == 'Sniper_Rifles':
        wclass[i] = 'Sniper Rifle'
    elif wclass[i] == 'Pistols':
        wclass[i] = 'Pistol'
    elif wclass[i] == 'Explosives':
        wclass[i] = 'Explosive'
    elif wclass[i] == 'Assault_Rifles':
        wclass[i] = 'Assault Rifle'
    elif wclass[i] == 'Machine_Guns':
        wclass[i] = 'Machine Gun'

                    
                        
#Create dictionary of main data
dat = {'Rarity': rare, 
       'Class': wclass,
       'Name': name,
       'Bullet Type': bt,
       'DPS': dps,
       'Damage': damage,
       'Fire Rate': fire,
       'Magazine Size': mag,
       'Reload Time': reload,
       'Structure Damage': sd}


#Dataframe
fortnite = pd.DataFrame(dat)

#Convert 9999 to NaN
fortnite[fortnite == 9999] = np.nan

#Replace 'Icon Series' in Rarity with 'Exotic'
fortnite = fortnite.replace(to_replace='Icon Series', value='Exotic')



#Scrape for chest probability
# url1 = urllib.request.urlopen( 'https://www.sportskeeda.com/esports/fortnite-data-miner-reveals-chest-loot-pool-works-chapter-2-season-5' )
# doc1 = url1.read()
# tree1 = BeautifulSoup( doc1 )

##
#Save as txt file to then read in to avoid any website updates
##

# url_get = requests.get('https://www.sportskeeda.com/esports/fortnite-data-miner-reveals-chest-loot-pool-works-chapter-2-season-5')
# soup = BeautifulSoup(url_get.content, 'lxml')

# with open("C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/fortnite_prob_html_17Jan2022.txt", 'w', encoding='utf-8') as f_out:
#     f_out.write(soup.prettify())


##
##Read in txt file to get HTML code from day it was pulled
##

hmtl_txt1 = open("C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/fortnite_prob_html_17Jan2022.txt", encoding='utf-8')
doc1 = hmtl_txt1.read()
tree1 = BeautifulSoup(doc1,"html.parser")


#Weapon list
wl = ['Assaults Rifles','Shotguns','SMGs','Pistols','Snipers']


prob = []
prob_name = []
class_prob = []
class_prob_name = []

links2 = tree1.find( id='article-content' )
links3 = links2.find_all( 'ul' )
a = -2
for m in links3:
    a += 1
    # print('start')
    # print(m)
    links4 = m.find_all( 'li' )
    for item in links4:
        if a == -1:
            prob_name.append('all')
            class_prob_name.append(item.string.partition('-')[0].strip())
            prob.append(item.string.strip())
            class_prob.append(item.string.partition('-')[2].strip())
        elif a >= 0 and a <= 4:
            prob_name.append(wl[a])
            prob.append(item.string.strip())
            
            
#Remove 'all' weapons probs from both lists
prob1 = prob[5:]
prob_name1 = prob_name[5:]    
    
    
    
#Create dictionary of class probabilites
class_dict = {'Class' : class_prob_name, 'Probability' : class_prob}

#Create dataframe of class probabilites
class_dat = pd.DataFrame(class_dict)

# Convet Probability to float   
for i in range(len(class_dat)):
    if class_dat['Probability'][i] != None:
        class_dat.loc[i,'Probability'] = float(class_dat.loc[i,'Probability'].replace('%',''))
    
#Clean up names so that we can merge onto data
for i in range(len(class_dat['Class'])):
    if class_dat.loc[i,'Class'] == 'SMG':
        class_dat.loc[i,'Class'] = 'Submachine Gun'
    elif class_dat.loc[i,'Class'] == 'Pistols':
        class_dat.loc[i,'Class'] = 'Pistol'
    if class_dat.loc[i,'Class'] == 'Snipers':
        class_dat.loc[i,'Class'] = 'Sniper Rifle'

    
    
    




#Open photo to read as bytes for Rekognition
photo = "C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/weapon_probs2.jpg"

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()
    
    
##
##
## Scrape diagram of chest probabilities
##
##

#Open AWS REKOGNITION
with open('C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Data/new_user_credentials.csv', 'r') as inp:
    next(inp)
    reader = csv.reader(inp)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

#Comment out Connection to AWS so I am not charged each time this runs
# '''
#Connect
client = boto3.client('rekognition',
                      aws_access_key_id = access_key_id,
                      aws_secret_access_key = secret_access_key, 
                      region_name = 'us-east-1')
# '''
#PNG of some weapon probs with my own labels added  

#Part 1 Image - 28 weapons
response1=client.detect_text(Image={'S3Object':{'Bucket':'fortniterekognition20220118','Name':'weapon_probs2_label_pt1.png'}})
# response=client.detect_text(Image={'Bytes': source_bytes})
rares = ['Blue', 'Green', 'Epic', 'Legendary']
iprob1=[]
iprob_name1=[]

for i in range(len(response1['TextDetections'])):
    # print(response1['TextDetections'][i]['DetectedText'])
    if '%' in response1['TextDetections'][i]['DetectedText']:
        iprob1.append(response1['TextDetections'][i]['DetectedText'])
    elif len(response1['TextDetections'][i]['DetectedText'].split()) > 1:
        # print('Adding this')
        iprob_name1.append(response1['TextDetections'][i]['DetectedText'])


#Somehow the probabilities are being read twice by Rekognition thus I will only use half of the list
iprob1 = iprob1[0:28]

#Clean up weapon names and strip off rarities
irare1 = []
for i in range(len(iprob_name1)):
    irare1.append(iprob_name1[i].replace('Green', 'Uncommon').replace('Blue','Rare').partition(' ')[0])
    iprob_name1[i] = iprob_name1[i].partition(' ')[2].replace('RTC', 'Rifle').replace('Boll','Bolt').replace('Plato','Pistol').replace('Pluto','Pistol').replace('Scan','Scar')
    if 'Tactical' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('Tactical','Tactical Shotgun')
    elif 'SMG' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('SMG','Submachine Gun')
    elif 'Pisto' in iprob_name1[i] and len(iprob_name1[i]) == 5:
        iprob_name1[i] = iprob_name1[i].replace('Pisto','Pistol')
    elif 'AR' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('AR','Assault Rifle')
    elif 'AR' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('AR','Assault Rifle')
    elif 'Lever Action' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('Lever Action','Lever Action Rifle')
    elif 'Lever SG' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('Lever SG','Lever Action Shotgun')
    elif 'Charge' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('Charge','Charge Shotgun')
    elif '-Action' in iprob_name1[i]:
        iprob_name1[i] = iprob_name1[i].replace('-Action','-Action Sniper Rifle')
        
        
        
#Part 2 image - 10
response2=client.detect_text(Image={'S3Object':{'Bucket':'fortniterekognition20220118','Name':'weapon_probs2_label_pt2.png'}})
# response=client.detect_text(Image={'Bytes': source_bytes})
# print(len(response1['TextDetections']))
rares = ['Blue', 'Green', 'Epic', 'Legendary']
iprob2=[]
iprob_name2=[]

#Clean up weapon names and strip off rarities
for i in range(len(response2['TextDetections'])):
    # print(response1['TextDetections'][i]['DetectedText'])
    if '%' in response2['TextDetections'][i]['DetectedText']:
        iprob2.append(response2['TextDetections'][i]['DetectedText'])
    elif len(response2['TextDetections'][i]['DetectedText'].split()) > 1:
        # print('Adding this')
        iprob_name2.append(response2['TextDetections'][i]['DetectedText'])

#Somehow the probabilities are being read twice by Rekognition thus I will only use half of the list
iprob2 = iprob2[0:10]

#Clean up weapon names and strip off rarities
irare2 = []
for i in range(len(iprob_name2)):
    irare2.append(iprob_name2[i].replace('Green', 'Uncommon').replace('Blue','Rare').partition(' ')[0])
    iprob_name2[i] = iprob_name2[i].partition(' ')[2].replace('SG','Shotgun').replace('Lever Shotgun','Lever Action Shotgun').replace('AR','Assault Rifle').replace('Breath','Breath Shotgun').replace('Scan','Scar').replace('Charge','Charge Shotgun').replace('Tactical','Tactical Shotgun')
    iprob2[i] = iprob2[i].replace('01%','0.1%')
    
    
#Concatenate data from both images
iprob = iprob1 + iprob2
iprob_name = iprob_name1 + iprob_name2
irare = irare1 + irare2


#Check for names that don't match main data
for n in iprob_name:
    if n not in name:
        print(n)
        
#Fix Scar and Compact Submachine Gun
for i in range(len(iprob_name)):
    if iprob_name[i] == 'Scar':
        iprob_name[i] = iprob_name[i].replace('Scar','Assault Rifle (SCAR)')
    elif iprob_name[i] == 'Compact Submachine Gun':
        iprob_name[i] = iprob_name[i].replace('Compact Submachine Gun','Compact SMG')
        


##
##
##
## Scrape stills from video
##
##

#Set up variable for data
vprob_name=[]
vprob=[]

#Function to get data from image in AWS S3
def get_image_data(image_name):
    
    #Image1 from video of more weapon probs
    response2=client.detect_text(Image={'S3Object':{'Bucket':'fortniterekognition20220118','Name':image_name}})
    
    print('Image: ' + image_name)
    
    #Get data from image
    if image_name not in  ['v_weapon4jpg.jpg','v_weapon7jpg.jpg']:
        for i in range(8):
            # print(response2['TextDetections'][i]['DetectedText'])
            
            if i<=3:
                vprob_name.append(response2['TextDetections'][i]['DetectedText'])
                print('Adding: ' + response2['TextDetections'][i]['DetectedText'] + ' to names')
            else:
                vprob.append(response2['TextDetections'][i]['DetectedText'])
                print('Adding: ' + response2['TextDetections'][i]['DetectedText'] + ' to probs')
    else:
        for i in range(2,10):
            # print(response2['TextDetections'][i]['DetectedText'])
            
            if i<=5:
                vprob_name.append(response2['TextDetections'][i]['DetectedText'])
                print('Adding: ' + response2['TextDetections'][i]['DetectedText'] + ' to names')
            else:
                vprob.append(response2['TextDetections'][i]['DetectedText'])
                print('Adding: ' + response2['TextDetections'][i]['DetectedText'] + ' to probs')

#Scrape all images
get_image_data('v_weapon1jpg.jpg')
get_image_data('v_weapon2jpg.jpg')
get_image_data('v_weapon3jpg.jpg')
get_image_data('v_weapon4jpg.jpg')
get_image_data('v_weapon5jpg.jpg')
get_image_data('v_weapon6jpg.jpg')
get_image_data('v_weapon7jpg.jpg')
get_image_data('v_weapon8jpg.jpg')
get_image_data('v_weapon9jpg.jpg')
get_image_data('v_weapon10jpg.jpg')
get_image_data('v_weapon11jpg.jpg')
get_image_data('v_weapon12jpg.jpg')





#Initialze lists for cleaning
vrare = []
vprob_name1 = []

#Clean up weapon names and strip off rarities
for i in range(len(vprob_name)):
    vrare.append(vprob_name[i].replace('Gray', 'Common').replace('Green', 'Uncommon').replace('Blue','Rare').partition(' ')[0])
    vprob_name[i] = (vprob_name[i].partition(' ')[2])


#Check for names not in main dataset
for n in vprob_name:
    if n not in name:
        print(n)

#Standardize names
for i in range(len(vprob_name)):
    if vprob_name[i] == 'Burst Rifle':
        vprob_name[i] = vprob_name[i].replace('Burst','Burst Assault')
    elif vprob_name[i] == 'SMG':
        vprob_name[i] = vprob_name[i].replace('SMG','Submachine Gun')
    elif vprob_name[i] == 'Dual Wield Pistols':
        vprob_name[i] = vprob_name[i].replace('Dual Wield','Dual')
    elif vprob_name[i] == 'Semi-Auto Sniper':
        vprob_name[i] = vprob_name[i].replace('Sniper','Sniper Rifle')
    elif vprob_name[i] == 'Scar Assault Rifle':
        vprob_name[i] = vprob_name[i].replace('Scar Assault Rifle','Assault Rifle (SCAR)')
    elif vprob_name[i] == 'Suppressed Scar':
        vprob_name[i] = vprob_name[i].replace('Suppressed Scar','Suppressed Assault Rifle')
    elif vprob_name[i] == 'Bolt Action Sniper':
        vprob_name[i] = vprob_name[i].replace('Bolt Action Sniper','Bolt-Action Sniper Rifle')
    elif vprob_name[i] == 'P90 SMG':
        vprob_name[i] = vprob_name[i].replace('P90','Compact')
    elif vprob_name[i] == 'Famas Assault Rifle':
        vprob_name[i] = vprob_name[i].replace('Famas','Burst')
    elif vprob_name[i] == 'Suppressed Scar Assault Rifle':
        vprob_name[i] = vprob_name[i].replace('Scar ','')

#ALl names match now!



#Now combine video probabilities and image probabilities
all_prob = iprob + vprob
all_name = iprob_name + vprob_name
all_rare = irare + vrare


#Check once more to make sure all names match
for n in all_name:
    if n not in name:
        print(n)

#All names check

#Create a dictionary to be converted into dataframe
prob_dict = {'Rarity' : all_rare,
            'Name' : all_name,
            'Probability' : all_prob}

#Create data frame
prob_dat = pd.DataFrame(prob_dict)

#Now check for duplicates across both data sources
new_all_name = []
delete_dups = []
keep_list = ['N' for i in range(len(all_name))]
for i in range(len(all_name)):
    test_name = all_rare[i] + ' ' + all_name[i]
    if test_name not in new_all_name:
        new_all_name.append(test_name)
        keep_list[i] = 'Y'
    elif test_name in new_all_name:
        delete_dups.append(test_name)
        
#See how many rows get deleted - 10
keep_list.count('N')

#Add Keep column to dataframe to then filter data
prob_dat['Keep'] = keep_list

#Filter out duplicates
prob_dat = prob_dat[prob_dat['Keep'] == 'Y']
    
#Drop keep flag
prob_dat = prob_dat.drop('Keep', axis=1)






## Now use some overly assumptuous mathematics to impute all other chest drop probabilites
## Use Bayes Rule here
## A = getting the weapon class (ex: SMG, AR, Sniper etc) (Also assuming SMG, AR< Sniper, Shotgun and Pistol equally likely - miniguns and launchers not included due to rarity)
## A1 = getting a specific weapon in the class A (assuming probability equal across all weapons in the class)
## B = opening a chest 
## P(A) = 1/6 (number of weapon classes being calculated: AR, Shotgun, SMG, Pistol, Sniper - excluding the minigun since we have all drop data on it)
## P(A1) = probability of specific weapon: calculated using  P(A)/(number of types of weapons in the class) * 2   (*2 because there are usually 2 of each weapon available each season)
## P(B) = 0.75 : I assume a player will find a chest in a match of Fortnite BR - if not you didn't pick a good drop 
## P(B|A) = probability of opening a chest with the weapon class (scraped from data above in class_dat DataFrame)
## P(B|A1) = probability of opening a chest with the specific weapon - calculated by dividing P(B|A)/(number weapons in class) 
## P(A1|B) = probability of finding the specific  weapon in a chest - this is what we want to solve for!
## Equation to sovle:
##                P(B|A1) x P(A1)
##    P(A1|B)   =  -------------
##                     P(B)


#Create probability columns in dataframe 

## P_A
class_dat['P_A'] = 1/5

## P_B
class_dat['P_B'] = 0.75

## P_A1
#Need number of each weapon in each class - use main data for this
wclass1 = set(wclass)

for i in wclass1:

    print(i)
    #P_A1
    class_dat.loc[class_dat.index[class_dat['Class'] == i], 'P_A1'] = class_dat[class_dat['Class'] == i]['P_A']/wclass.count(i) * 2
    
    #P(B|A)
    class_dat.loc[class_dat.index[class_dat['Class'] == i], 'P_BA'] = class_dat[class_dat['Class'] == i]['Probability']
    
    #P(B|A1)
    class_dat.loc[class_dat.index[class_dat['Class'] == i], 'P_BA1'] = class_dat[class_dat['Class'] == i]['Probability']/wclass.count(i) *2


#P(A1|B)
class_dat['C_Probability'] = class_dat['P_BA1'] * class_dat['P_A1'] / class_dat['P_B'] * 100



#Join scraped chest probability onto main data by rarity and name
all_fort = sqldf("""
                      select a.*, b.Probability from fortnite as a
                          left join prob_dat as b
                          on a.Rarity = b.Rarity and a.Name = b.Name;
      
                """)
                

#Convert Probability to a numeric var
for i in range(len(all_fort)):
    if all_fort['Probability'][i] != None:
        all_fort.loc[i,'Probability'] = float(all_fort.loc[i,'Probability'].replace('%',''))


#By nature of the game, all Exotic, Mythic and Transcendent weapons do not appear in chests
#thus set probability to 0 for these. Also set other weapons not available from chest
no_chest = ['Stark Industries Energy Rifle']
for i in range(len(all_fort)):
    if all_fort.loc[i,'Rarity'] in ['Exotic', 'Mythic', 'Transcendent']:
        all_fort.loc[i,'Probability'] = 0.0
    elif all_fort.loc[i,'Name'] in no_chest:
        all_fort.loc[i,'Probability'] = 0.0
    # print(type(all_fort['Probability'][i]))
    # print(all_fort['Probability'][i])
    # elif all_fort.loc[i,'Probability'] == None:
    #     all_fort.loc[i,'Probability'] == float("nan")


#Change None to 9999.9
all_fort.Probability.fillna(value=9999.9, inplace = True)
# all_fort.Probability.fillna(value=np.nan, inplace = True)


freqs = all_fort[['Probability','Class']].value_counts()
all_fort['Probability'].value_counts()

## For any weapons missing chest drop probability, use the "imputed" drop calcualted about in class_dat

#Join on class
all_fort2 = sqldf("""

                      select a.Rarity, a.Class, a.Name, a.'Bullet Type', a.DPS, a.Damage, a.'Fire Rate',
                             a.'Magazine Size', a.'Reload Time', a.'Structure Damage', b.C_Probability as 'Probability'
                          from ( select * 
                                 from all_fort
                                 where Probability = 9999.9 and (Class not in  ('Explosive', 'Machine Gun'))) as a
                          inner join class_dat as b
                          on a.Class = b.Class
                          
                          union 
                          
                          select * 
                              from all_fort
                              where Probability != 9999.9 or (Class in ('Explosive', 'Machine Gun')) 
                      
                """)


#Replace 9999.9 with nan
all_fort2[all_fort2 == 9999.9] = np.nan


   
    




#Rename data
fortnite2 = all_fort2



#Impute missing values using KNN
imputer = KNNImputer(n_neighbors=2, weights="uniform")
fortnite2[['DPS', 'Damage', 'Fire Rate', 'Magazine Size', 'Reload Time', 'Structure Damage','Probability']] = imputer.fit_transform(fortnite2[['DPS', 'Damage', 'Fire Rate', 'Magazine Size', 'Reload Time', 'Structure Damage','Probability']])

#Convert Probability column to true probability not percentage
fortnite2['Probability'] = fortnite2['Probability'] / 100

#Standardize 'Bullet Type'
for i in range(len(fortnite2)):
    if 'slugs' in fortnite2.loc[i,'Bullet Type']:
        fortnite2.loc[i,'Bullet Type'] = fortnite2['Bullet Type'][i].replace('slugs','Slugs')
    elif 'bullets' in fortnite2.loc[i,'Bullet Type']:
        fortnite2.loc[i,'Bullet Type'] = fortnite2['Bullet Type'][i].replace('bullets','Bullets')
        
        
#Check Bullet Type now
fortnite2['Bullet Type'].value_counts()
        
#Check Rarity now
fortnite2['Rarity'].value_counts()


#Give ID values to each row for easy sampling purposes
fortnite2['id'] = fortnite2.index



## Posture data for sklearn

#Dummy code 'Bullet Type' variable
fortnite3 = pd.get_dummies(fortnite2, prefix='Bullet', columns=['Bullet Type'], drop_first=True).sort_values('Class')
fortnite3 = pd.get_dummies(fortnite3, prefix='Class', columns=['Class'], drop_first=True)
fortnite3['Class'] = fortnite2['Class']

#Output data as csv
# fortnite2.to_csv('C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Output/fortnite_final_no_dummies.csv')
# fortnite3.to_csv('C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Output/fortnite_final_dummies.csv')
# fortnite4.to_csv('C:/Users/Richard Pincus/Documents/Classes - MSA/Side Project/Fortnite/Output/fortnite_final_20220316.csv')


## Stratified Random Sampling to get train/test split and ensure proportions are ok

#Check population proportions
fortnite3.value_counts('Class') / len(fortnite3)

#Sample 70% for train
random.seed(101)
train = fortnite3.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.7))

#Check sample proportions
train.value_counts('Class') / len(train) #looks great

#Get test data
all_col_names = list(fortnite3.axes[1])
temp1 = fortnite3.merge(train, how='left', on=all_col_names, indicator=True)
test = temp1.loc[temp1['_merge'] == 'left_only',all_col_names]

#Ensure no data is left behind
len(train) + len(test) == len(fortnite3) #clear for take off





#Get target variable
Y_train = train['Rarity']
Y_test = test['Rarity']

#Convert target to numeric
Y_train = Y_train.replace('Common',0).replace('Uncommon',1).replace('Rare',2).replace('Epic',3).replace('Legendary',4).replace('Mythic',5).replace('Exotic',6).replace('Transcendent',7)
Y_test = Y_test.replace('Common',0).replace('Uncommon',1).replace('Rare',2).replace('Epic',3).replace('Legendary',4).replace('Mythic',5).replace('Exotic',6).replace('Transcendent',7)

#Get predictors - remove DPS due to mutlicollinearity with many other vars
X_train = train[['Damage','Fire Rate','Magazine Size','Reload Time','Structure Damage','Probability','Bullet_Light Bullets','Bullet_Medium Bullets','Bullet_Rockets',"Bullet_Shells 'n' Slugs",'Class_Explosive','Class_Machine Gun','Class_Pistol','Class_Shotgun','Class_Sniper Rifle','Class_Submachine Gun']]
X_test = test[['Damage','Fire Rate','Magazine Size','Reload Time','Structure Damage','Probability','Bullet_Light Bullets','Bullet_Medium Bullets','Bullet_Rockets',"Bullet_Shells 'n' Slugs",'Class_Explosive','Class_Machine Gun','Class_Pistol','Class_Shotgun','Class_Sniper Rifle','Class_Submachine Gun']]



#Start with a Random Forest Classification because this will be the simplest
rf = RandomForestClassifier(n_estimators=16, random_state=101)
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)

print( 'Random Forest Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_rf ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( rf.score( X_train, Y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( rf.score( X_test, Y_test ) ) )
# print( 'Out-of-bag accuracy: {:.2f} (out of 1)'.format( rf.oob_score_ ) )




#Build Neural Network
ppn = Perceptron( max_iter=100, eta0=0.9, random_state=101 )
ppn.fit( X_train, Y_train )
y_pred_nn = ppn.predict( X_test )

# Print results of classification
print( 'Perceptron Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_rf ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( ppn.score( X_train, Y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( ppn.score( X_test, Y_test ) ) )




#Build XGBoost tree
xgb_model = xgb.XGBClassifier(objective="multi:softprob", use_label_encoder=False, random_state=101)
xgb_model.fit(X_train, Y_train)
y_pred_xgb = xgb_model.predict(X_test)
confusion_matrix(Y_test, y_pred_xgb)

print( 'XGBoost Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_xgb ).sum() )
# print( 'Accuracy on training data: {:.2f} (out of 1)'.format( xgb.score( X_train, Y_train ) ) )
# print( 'Accuracy on test data: {:.2f} (out of 1)'.format( xgb.score( X_test, Y_test ) ) )


#Build Multinomial Logistic Regression
mlogit_model = LogisticRegression(random_state=101, max_iter = 10000)
mlogit_model.fit(X_train, Y_train)
y_pred_mlogit = mlogit_model.predict(X_test)

# Print results of classification
print( 'Logistic Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_mlogit ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( mlogit_model.score( X_train, Y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( mlogit_model.score( X_test, Y_test ) ) )


#Predictions Dictionary
pred_dict = {'RF': y_pred_rf,
             'NN': y_pred_nn,
             'XGB': y_pred_xgb,
             'MLog': y_pred_mlogit}

predictions = pd.DataFrame(pred_dict, index=Y_test.index, columns = ['RF','NN','XGB','MLog'])

#Ensemble predictions
predictions['Avg'] = predictions.mean(axis = 1).round().astype(int)


print( 'Ensembled Results:' )
print( 'Misclassified samples: %d' % ( Y_test != predictions['Avg'] ).sum() )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format((1 - (( Y_test != predictions['Avg'] ).sum() ) / len(Y_test))))









#I certainly need more observations for these ML models to be able to classify correctly
#I shall impute Synthetic Data usling IMBLearn - can't get SMOTE to run use classic upsampling here




'''

#First Miniguns only have 4 observations, this will disrupt the SMOTE algorithm. 
#I need to oversample Miniguns in train to double this sample size
miniguns = pd.DataFrame(train[train['Class'] == 'Machine Gun'])
idx = miniguns.index
new_miniguns = pd.DataFrame().reindex_like(miniguns).reset_index(drop=True)

i = -1
random.seed(101)
while len(new_miniguns) < 5:
    j = random.sample(list(idx), 1)[0]
    if random.random() <= 0.5:
        print('next')
    else:
        # print('sampling')
        # print(j)
        i += 1
        new_miniguns.loc[i] = miniguns.loc[j]


#Exotic and Transcendent weapons have < 6 obs, need to upsample these from SMOTE as well

#Exotic
exotic = pd.DataFrame(train[train['Rarity'] == 'Exotic'])
idx = exotic.index
new_exotic = pd.DataFrame().reindex_like(exotic).reset_index(drop=True)

if len(exotic) < 6:
    print('Not enough exotic weapons, must upsample before SMOTE')
    i = -1
    random.seed(101)
    while len(new_exotic) < 7:
        j = random.sample(list(idx), 1)[0]
        if random.random() <= 0.5:
            print('next')
        else:
            # print('sampling')
            # print(j)
            i += 1
            new_exotic.loc[i] = exotic.loc[j]
        
        
#Transcendent - only 1 in training but if seed ever changes there could be multiple in seed to use same code
tran = pd.DataFrame(train[train['Rarity'] == 'Transcendent'])
idx = tran.index
new_tran = pd.DataFrame().reindex_like(tran).reset_index(drop=True)

if len(tran) < 6:
    print('Not enough transcendent weapons, must upsample before SMOTE')
    i = -1
    random.seed(101)
    while len(new_tran) < 6:
        j = random.sample(list(idx), 1)[0]
        if random.random() <= 0.5:
            print('next')
        else:
            # print('sampling')
            # print(j)
            i += 1
            new_tran.loc[i] = tran.loc[j]


#Append new_miniguns, new_exotic and new_tran data to train
if len(exotic) < 6 and len(tran) < 6: #append all sampled guns
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_exotic, ignore_index=True).append(new_tran, ignore_index=True)
    
elif len(exotic) < 6: #append exotic and not transcendent
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_exotic, ignore_index=True)
    
elif len(tran) < 6: #append transcendent and not exotic
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_tran, ignore_index=True)

else: #only append miniguns
    train_adj = train.append(new_miniguns, ignore_index=True)


#Get predictors again - remove DPS due to mutlicollinearity with many other vars
X_train_adj = train_adj[['Damage','Fire Rate','Magazine Size','Reload Time','Structure Damage','Probability','Bullet_Light Bullets','Bullet_Medium Bullets','Bullet_Rockets',"Bullet_Shells 'n' Slugs"]]

#Get target variable
Y_train_adj = train_adj['Rarity']

#Convert target to numeric
Y_train_adj = Y_train_adj.replace('Common',0).replace('Uncommon',1).replace('Rare',2).replace('Epic',3).replace('Legendary',4).replace('Mythic',5).replace('Exotic',6).replace('Transcendent',7)



#Gerneate synthetic data using SMOTENC
smotenc1 = SMOTENC(categorical_features=[6,7,8,9], random_state=101)
X_resampled, Y_resampled = smotenc1.fit_resample(X_train_adj, Y_train_adj)
#This data is definitely synthetic, but none of the synthesized rows have bullet types which is going to be an issue



# pd.crosstab(X_train_adj["Bullet_Shells 'n' Slugs"], columns = 'count')
# X_train_adj.columns
# Y_train_adj.value_counts()

'''

'''
#I want to upsample using SMOTE but need to ensure each weapon has a bullet type
#I need to do all of the above resampling on fortnite2 without the dummy vars first

#Get same training sample from forntite2
train2 = sqldf("""
               select a.* 
                   from fortnite2 as a
                   inner join train as b
                   on a.id = b.id;
               """)
               
               
               
#First Miniguns only have 4 observations, this will disrupt the SMOTE algorithm. 
#I need to oversample Miniguns in train to double this sample size
miniguns = pd.DataFrame(train2[train2['Class'] == 'Machine Gun'])
idx = miniguns.index
new_miniguns = pd.DataFrame().reindex_like(miniguns).reset_index(drop=True)

i = -1
random.seed(101)
while len(new_miniguns) < 5:
    j = random.sample(list(idx), 1)[0]
    if random.random() <= 0.5:
        print('next')
    else:
        print('sampling')
        print(j)
        i += 1
        new_miniguns.loc[i] = miniguns.loc[j]


#Exotic and Transcendent weapons have < 6 obs, need to upsample these from SMOTE as well

#Exotic
exotic = pd.DataFrame(train2[train2['Rarity'] == 'Exotic'])
idx = exotic.index
new_exotic = pd.DataFrame().reindex_like(exotic).reset_index(drop=True)

i = -1
random.seed(101)
while len(new_exotic) < 5:
    j = random.sample(list(idx), 1)[0]
    if random.random() <= 0.5:
        print('next')
    else:
        print('sampling')
        print(j)
        i += 1
        new_exotic.loc[i] = exotic.loc[j]
        
        
#Transcendent - only 1 in training but if seed ever changes there could be multiple in seed to use same code
tran = pd.DataFrame(train2[train2['Rarity'] == 'Transcendent'])
idx = tran.index
new_tran = pd.DataFrame().reindex_like(tran).reset_index(drop=True)

i = -1
random.seed(101)
while len(new_tran) < 6:
    j = random.sample(list(idx), 1)[0]
    if random.random() <= 0.5:
        print('next')
    else:
        print('sampling')
        print(j)
        i += 1
        new_tran.loc[i] = tran.loc[j]


#Append new_miniguns, new_exotic and new_tran data to train
train2_adj = train2.append(new_miniguns, ignore_index=True).append(new_exotic, ignore_index=True).append(new_tran, ignore_index=True)



#Get predictors again - remove DPS due to mutlicollinearity with many other vars
X_train2_adj = train2_adj[['Damage','Fire Rate','Magazine Size','Reload Time','Structure Damage','Probability','Bullet Type']]

#Get target variable
Y_train2_adj = train2_adj['Rarity']

#Convert target to numeric
Y_train2_adj = Y_train2_adj.replace('Common',0).replace('Uncommon',1).replace('Rare',2).replace('Epic',3).replace('Legendary',4).replace('Mythic',5).replace('Exotic',6).replace('Transcendent',7)


#Gerneate synthetic data using SMOTENC
smotenc1 = SMOTENC(categorical_features=[6], random_state=101)
X_resampled, Y_resampled = smotenc1.fit_resample(X_train2_adj, Y_train2_adj)
'''





#Upsample miniguns, exotic weapons, and transcendent weapons as needed



#Check sample proportions
train.value_counts('Class') / len(train) #machine guns need a boost
train.value_counts('Rarity') / len(train) #exotic and transcendent need a boost


#First Miniguns only have 4 observations 
#I need to oversample Miniguns in train to double this sample size
miniguns = pd.DataFrame(train[train['Class'] == 'Machine Gun'])
idx = miniguns.index
new_miniguns = pd.DataFrame().reindex_like(miniguns).reset_index(drop=True)

i = -1
random.seed(101)
while len(new_miniguns) < 10:
    j = random.sample(list(idx), 1)[0]
    if random.random() <= 0.5:
        print('next')
    else:
        # print('sampling')
        # print(j)
        i += 1
        new_miniguns.loc[i] = miniguns.loc[j]


#If Exotic and Transcendent weapons have < 6 obs

#Exotic
exotic = pd.DataFrame(train[train['Rarity'] == 'Exotic'])
idx = exotic.index
new_exotic = pd.DataFrame().reindex_like(exotic).reset_index(drop=True)

if len(exotic) < 6:
    print('Not enough exotic weapons, must upsample before SMOTE')
    i = -1
    random.seed(101)
    while len(new_exotic) < 10:
        j = random.sample(list(idx), 1)[0]
        if random.random() <= 0.5:
            print('next')
        else:
            # print('sampling')
            # print(j)
            i += 1
            new_exotic.loc[i] = exotic.loc[j]
        
        
#Transcendent - only 1 in training but if seed ever changes there could be multiple in seed to use same code
tran = pd.DataFrame(train[train['Rarity'] == 'Transcendent'])
idx = tran.index
new_tran = pd.DataFrame().reindex_like(tran).reset_index(drop=True)

if len(tran) < 6:
    print('Not enough transcendent weapons, must upsample before SMOTE')
    i = -1
    random.seed(101)
    while len(new_tran) < 12:
        j = random.sample(list(idx), 1)[0]
        if random.random() <= 0.5:
            print('next')
        else:
            # print('sampling')
            # print(j)
            i += 1
            new_tran.loc[i] = tran.loc[j]


#Append new_miniguns, new_exotic and new_tran data to train
if len(exotic) < 6 and len(tran) < 6: #append all sampled guns
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_exotic, ignore_index=True).append(new_tran, ignore_index=True)
    
elif len(exotic) < 6: #append exotic and not transcendent
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_exotic, ignore_index=True)
    
elif len(tran) < 6: #append transcendent and not exotic
    train_adj = train.append(new_miniguns, ignore_index=True).append(new_tran, ignore_index=True)

else: #only append miniguns
    train_adj = train.append(new_miniguns, ignore_index=True)


#Get predictors again - remove DPS due to mutlicollinearity with many other vars
X_train_adj = train_adj[['Class','Damage','Fire Rate','Magazine Size','Reload Time','Structure Damage','Probability','Bullet_Light Bullets','Bullet_Medium Bullets','Bullet_Rockets',"Bullet_Shells 'n' Slugs",'Class_Explosive','Class_Machine Gun','Class_Pistol','Class_Shotgun','Class_Sniper Rifle','Class_Submachine Gun']]

#Get target variable
Y_train_adj = train_adj['Rarity']

#Convert target to numeric
Y_train_adj = Y_train_adj.replace('Common',0).replace('Uncommon',1).replace('Rare',2).replace('Epic',3).replace('Legendary',4).replace('Mythic',5).replace('Exotic',6).replace('Transcendent',7)




#Check sample proportions
train_adj.value_counts('Class') / len(train_adj) #machine guns need a boost
train_adj.value_counts('Rarity') / len(train_adj) #exotic and transcendent need a boost



#Now attempt SMOTENC

#Gerneate synthetic data using SMOTENC
smotenc1 = SMOTENC(categorical_features=[0,7,8,9,10,11,12,13,14,15,16], random_state=101)
X_resampled, Y_resampled = smotenc1.fit_resample(X_train_adj, Y_train_adj)



#Check sample proportions
X_resampled.value_counts('Class') / len(X_resampled) #machine guns need a boost
Y_resampled.value_counts('Rarity') / len(Y_resampled) #exotic and transcendent need a boost






#Remove class variable from training data
X_resampled = X_resampled.drop(labels="Class", axis=1)




#Start with a Random Forest Classification because this will be the simplest
rf = RandomForestClassifier(n_estimators=16, random_state=101)
rf.fit(X_resampled, Y_resampled)
y_pred_rf = rf.predict(X_test)

print( 'Random Forest Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_rf ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( rf.score( X_resampled, Y_resampled ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( rf.score( X_test, Y_test ) ) )
# print( 'Out-of-bag accuracy: {:.2f} (out of 1)'.format( rf.oob_score_ ) )




#Build Neural Network
ppn = Perceptron( max_iter=100, eta0=0.9, random_state=101 )
ppn.fit( X_resampled, Y_resampled )
y_pred_nn = ppn.predict( X_test )

# Print results of classification
print( 'Perceptron Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_rf ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( ppn.score( X_resampled, Y_resampled ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( ppn.score( X_test, Y_test ) ) )




#Build XGBoost tree
xgb_model = xgb.XGBClassifier(objective="multi:softprob", use_label_encoder=False, random_state=101)
xgb_model.fit(X_resampled, Y_resampled)
y_pred_xgb = xgb_model.predict(X_test)
confusion_matrix(Y_test, y_pred_xgb)

print( 'XGBoost Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_xgb ).sum() )
# print( 'Accuracy on training data: {:.2f} (out of 1)'.format( xgb.score( X_train, Y_train ) ) )
print( 'Accuracy on test data: %.2f (out of 1)' % (( Y_test == y_pred_xgb ).sum() / len(Y_test)) )


#Build Multinomial Logistic Regression
mlogit_model = LogisticRegression(random_state=101, max_iter = 10000)
mlogit_model.fit(X_resampled, Y_resampled)
y_pred_mlogit = mlogit_model.predict(X_test)

# Print results of classification
print( 'Logistic Model:' )
print( 'Misclassified samples: %d' % ( Y_test != y_pred_mlogit ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( mlogit_model.score( X_resampled, Y_resampled ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( mlogit_model.score( X_test, Y_test ) ) )


#Predictions Dictionary
pred_dict = {'RF': y_pred_rf,
             'NN': y_pred_nn,
             'XGB': y_pred_xgb,
             'MLog': y_pred_mlogit}

predictions = pd.DataFrame(pred_dict, index=Y_test.index, columns = ['RF','NN','XGB','MLog'])

#Ensemble predictions
predictions['Avg'] = predictions.mean(axis = 1).round().astype(int)


print( 'Ensembled Results:' )
print( 'Misclassified samples: %d' % ( Y_test != predictions['Avg'] ).sum() )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format((1 - (( Y_test != predictions['Avg'] ).sum() ) / len(Y_test))))






#XGB is the winner in predicting Rarity at 33% accuracy.
#I feel as though more data could greatly increase these ML models and the accuracy of predictions.




