#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imageScrape import img_scrape
import os, errno

#List of bad dogs, see the python notebook for details
#Criterion: using a threshold of classification < 0.96

bad_dogs = {
'appenzeller',
'black-and-tan_coonhound',
'boston_bull' 'brabancon_griffon',
'brittany_spaniel',
'bull_mastiff',
'chihuahua',
'collie',
'dingo',
'doberman',
'english_foxhound',
'entlebucher',
'eskimo_dog', #happens to have low amount of images, coincidence?
'flat-coated_retriever',
'giant_schnauzer',
'greater_swiss_mountain_dog',
'irish_terrier',
'italian_greyhound' ,
'japanese_spaniel' ,
'lakeland_terrier',
'lhasa',
'miniature_poodle',
'miniature_schnauzer',
'redbone' ,
'rhodesian_ridgeback',
'scottish_deerhound',
'siberian_husky',
'staffordshire_bullterrier',
'standard_schnauzer',
'toy_poodle',
'vizsla',
'walker_hound',
'whippet',
'wire-haired_fox_terrier',
'yorkshire_terrier'}


#Several preset constant
num_images = 20
#ADJUST TO YOUR PREFERRED DIRECTORY!!!
save_dir = 'C:/Users/Stephan/Desktop/Bad_Dogs'

def create_dirs(query,dir):
	"""
	create_dirs(QUERY)
	Creates directory in the parent folder specified in SAVE_DIR with the name in QUERY
	"""
	
	directory = os.path.dirname(dir)
	try:
		os.makedirs(dir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


if __name__ == '__main__':
	
	for dogs in set(bad_dogs):
		#Call create_dirs, create directories for each dog
		dir = f'{save_dir}/{dogs}'
		create_dirs(f'{dogs}',dir)
		
		#when directories are created,  then call imagescrape
		# and put dogs in their respective directories
		
		img_scrape(f'{dogs}', num_images, dir)