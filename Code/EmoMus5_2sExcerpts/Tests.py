# fMRI Sonification classes
import fMRI_analysis as fa

# Standard library imports
import numpy as num
import cPickle as pickle
import os, glob
import xlwt

import xml.etree.ElementTree as ET

class EmoMus5_Test():
    def __init__(self):
        os.chdir('../../Data/EmoMus5')
        self.dir = os.getcwd()
        self.data_path = self.dir + "/Data/"
        self.info_path = self.dir + "/Info/"

    def start(self):
        # Generate some data to plot:

        print "\n---- STARTING EMOMUS5 TESTS ---"

        print "\nEmoMus5_Test_2s -> Description"
        print "EmoMus5_Test_2s -> There is one audio descriptor per scan (every 2 seconds)"
        print "EmoMus5_Test_2s -> Reading data files from " + self.data_path
        
        self.createResultsExcelFile()
        self.readScansOrder()
        self.startTestForEachUser()
           
        print "\n---- END EMOMUS5 TESTS ---"
        print "---- WELL DONE ---"

    def createResultsExcelFile(self):

        results_path = self.dir + '/ResultsPM_EmoMus5.xls'
        print "\nEmoMus5_Test_2s -> Creating the results excel file. "
        print "EmoMus5_Test_2s -> All the prediction model results will be saved at " + results_path

        wb = xlwt.Workbook()
        ws = wb.add_sheet('First Sheet')
        wb.save(results_path)


    def readScansOrder(self):
        
        data_xml_path = self.info_path + "EmoMus5.xml"

        print "\nEmoMus5_Test_2s -> Scan order"
        print "EmoMus5_Test_2s -> Reading fMRI scan order from: " + data_xml_path
        print "EmoMus5_Test_2s -> This is the order that the musical excerpts were presented to all participants"
        print "EmoMus5_Test_2s -> In this experiment each excerpt was presented twice. Each presentation lasted 30 seconds"
        print "EmoMus5_Test_2s -> After each excerpt there where 18 seconds without stimulation"
        print "EmoMus5_Test_2s -> There are three types of emotions attached to the musical excerpts: Happy, Anxious and Neutral"
           
        tree = ET.parse(data_xml_path)
        root = tree.getroot()

        self.scansOrder = []
        scans = []
        for child in root:
            for scan in child.findall('scan'):
                onset = int(scan.find('onset').text)
                id = scan.get('id')
                #print id, onset
                scans.append(onset)
            
            print "EmoMus5_Test_2s -> Emotion: "+ child.tag + " Onset start scan order: " + str(scans)
           
            self.scansOrder.append(scans)
            scans = []

    def startTestForEachUser(self):

        print "\nEmoMus5_Test_2s -> Starting analysis for each user"

        usersDir = glob.glob( os.path.join(self.data_path, '*.nii.gz'))

        for userNum, userPath in enumerate(usersDir):
            
            fileName = userPath.split("/")[-1]
            user = fileName.split(".")[0]

            print "EmoMus5_Test_2s -> Test Number: " + " (" + str(userNum+1) + "/" + str(len(usersDir))+ ")"
            print "EmoMus5_Test_2s -> User Name: "+ str(user) 
            
            fMRI = fa.fMRI_analysis(self.dir, user)  
            fMRI.readData() 
            
            try:
                selPath = self.info_path + "selectedFeatures_" + user + ".dat"
                fi = open(selPath, 'r')
                print "\nEmoMus5_Test_2s -> Opening previously calculated selected features from " + selPath
            except IOError:
                #No such file
                selPath = self.info_path + "selectedFeatures_" + user + ".dat"
                print "EmoMus5_Test_2s -> No such file: " + selPath
                print "EmoMus5_Test_2s -> Calculating selected features and saving them to " + selPath
                (self.ANOVA_ind,self.emotions) = fMRI.selectFeatures(self.scansOrder)
            else:
                print "EmoMus5_Test_2s -> Loading selected features " 
                self.ANOVA_ind = pickle.load(fi) # ANOVA analysis best voxels indexes
                print "EmoMus5_Test_2s -> Loading ANOVA analysis best voxel indexes with vector size of ", len(self.ANOVA_ind)
                self.emotions = pickle.load(fi)
                print "EmoMus5_Test_2s -> Loading fMRI scan indexes sorted to the different emotion music groups"
                #print "EmoMus5_Test_2s -> Emotion: Anxious " + str(self.emotions[0])
                #print "EmoMus5_Test_2s -> Emotion: Happy " + str(self.emotions[1])
                #print "EmoMus5_Test_2s -> Emotion: Neutral " + str(self.emotions[2])

                fi.close()
                
            fMRI.predictionModel(self.ANOVA_ind,self.emotions)


