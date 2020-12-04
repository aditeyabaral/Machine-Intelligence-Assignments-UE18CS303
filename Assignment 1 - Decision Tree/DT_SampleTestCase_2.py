from Assignment1 import *

def test_case():
    print("Testing with file Test.csv\n")
    df = pd.read_csv('Test.csv')
    #print(df)
    print('Dataset entropy : ',np.isclose(0.9709505944546686,get_entropy_of_dataset(df)))  #0.9709505944546686
    print('Sky entropy : ',np.isclose(0.9509775004326937,get_entropy_of_attribute(df, 'Sky'))) #0.9509775004326937
    print('Sky IG : ', np.isclose(0.01997309402197489, get_information_gain(df, 'Sky'))) #0.01997309402197489
    print('Airtemp entropy : ',np.isclose(0.6490224995673063, get_entropy_of_attribute(df, 'Airtemp'))) #0.6490224995673063
    print('Airtemp IG : ', np.isclose(0.3219280948873623, get_information_gain(df, 'Airtemp'))) #0.3219280948873623
    print('Humidity entropy : ',np.isclose(0.9509775004326937, get_entropy_of_attribute(df, 'Humidity'))) #0.9509775004326937
    print('Humidity IG : ', np.isclose(0.01997309402197489, get_information_gain(df, 'Humidity'))) #0.01997309402197489
    print('Water entropy : ',np.isclose(0.8, get_entropy_of_attribute(df, 'Water'))) #0.8
    print('Water IG : ', np.isclose(0.17095059445466854,get_information_gain(df, 'Water'))) #0.17095059445466854
    print('Forecast entropy : ',np.isclose(0.9509775004326937, get_entropy_of_attribute(df, 'Forecast'))) #0.9509775004326937
    print('Forecast IG : ', np.isclose(0.01997309402197489, get_information_gain(df, 'Forecast'))) #0.01997309402197489
    print('Attr to split: ', get_selected_attribute(df)[1]=='Airtemp') #Airtemp
    
    
    df = pd.read_csv('Test1.csv')
    print("\nTesting with file Test1.csv\n")
    print('Dataset entropy : ', np.isclose(0.9402859586706311, get_entropy_of_dataset(df)))  #0.9402859586706311
    print('Age entropy : ', np.isclose(0.6324823551623816, get_entropy_of_attribute(df, 'Age'))) #0.6324823551623816
    print('Age IG : ', np.isclose(0.30780360350824953, get_information_gain(df, 'Age'))) #0.30780360350824953
    print('Income entropy : ', np.isclose(0.9110633930116763, get_entropy_of_attribute(df, 'Income'))) #0.9110633930116763
    print('Income IG : ', np.isclose(0.02922256565895487, get_information_gain(df, 'Income'))) #0.02922256565895487
    print('Student entropy : ', np.isclose(0.7884504573082896, get_entropy_of_attribute(df, 'Student'))) #0.7884504573082896
    print('Student IG : ', np.isclose(0.15183550136234159, get_information_gain(df, 'Student'))) #0.15183550136234159
    print('Credit_rating entropy : ', np.isclose(0.8921589282623617, get_entropy_of_attribute(df, 'Credit_rating'))) #0.8921589282623617
    print('Credit_rating IG : ', np.isclose(0.04812703040826949, get_information_gain(df, 'Credit_rating'))) #0.04812703040826949
    print('Attr to split: ', get_selected_attribute(df)[1]=='Age') #Age
    
    
    print("\nTesting with file Test2.csv\n")
    df = pd.read_csv('Test2.csv')
    print('Dataset entropy : ', np.isclose(0.9852281360342515, get_entropy_of_dataset(df))) #0.9852281360342515
    print('Salary entropy : ', np.isclose(0.5156629249195446, get_entropy_of_attribute(df,'salary'))) #0.5156629249195446
    print('Salary IG : ', np.isclose(0.46956521111470695, get_information_gain(df,'salary'))) #0.46956521111470695
    print('Location entropy : ', np.isclose(0.2857142857142857, get_entropy_of_attribute(df,'location'))) #0.2857142857142857
    print('Location IG : ', np.isclose(0.6995138503199658, get_information_gain(df,'location'))) #0.6995138503199658
    print('Attr to split:', get_selected_attribute(df)[1]=='location')
    

    print("\nTesting with file Test3.csv\n")
    df = pd.read_csv('Test3.csv')
    print('Dataset entropy : ', np.isclose(0.9709505944546686, get_entropy_of_dataset(df))) #0.9709505944546686
    print('Toothed entropy : ', np.isclose(0.963547202339972, get_entropy_of_attribute(df,'toothed'))) #0.963547202339972
    print('Toothed IG : ', np.isclose(0.007403392114696539, get_information_gain(df,'toothed'))) #0.007403392114696539
    print('Breathes entropy : ', np.isclose(0.8264662506490407, get_entropy_of_attribute(df,'breathes'))) #0.8264662506490407
    print('Breathes IG : ', np.isclose(0.1444843438056279, get_information_gain(df,'breathes'))) #0.1444843438056279
    print('Legs entropy : ', np.isclose(0.4141709450076292, get_entropy_of_attribute(df,'legs'))) #0.4141709450076292
    print('Legs IG : ', np.isclose(0.5567796494470394, get_information_gain(df,'legs'))) #0.5567796494470394
    print('Attr to split:', get_selected_attribute(df)[1] == 'legs') #legs

    print("\nTesting with file Test5.csv\n")
    df = pd.read_csv("Test5.csv")
    print("Dataset Entropy: ", np.isclose(get_entropy_of_dataset(df), 1.7295739585136223))
    print("Entropy of category attribute:", np.isclose(get_entropy_of_attribute(df, "category"), 0.9182958340544896))
    print("Category IG: ", np.isclose(get_information_gain(df, 'category'), 0.811278124459))
    
    print("\nTesting with file Test6.csv\n")
    df = pd.read_csv("Test6.csv")
    print("Dataset Entropy: ", np.isclose(get_entropy_of_dataset(df), 0.9709505944546686))
    print("Color entropy: ", np.isclose(get_entropy_of_attribute(df, 'color'),0.9709505944546686))
    print("Color IG:", np.isclose(0.0, get_information_gain(df, 'color')))
    print("Size entropy:", np.isclose(0.9709505944546686, get_entropy_of_attribute(df, 'size')))
    print("Size IG:", np.isclose(0.0, get_information_gain(df, 'size')))
    print("act entropy:",np.isclose(get_entropy_of_attribute(df, 'act'),0.5509775004326937))
    print("Act IG: ",np.isclose(0.4199730940219749, get_information_gain(df, 'act')))
    print("age Entropy:", np.isclose(0.5509775004326937, get_entropy_of_attribute(df, 'age')))
    print("age IG:", np.isclose(get_information_gain(df, 'age'),0.4199730940219749))

    print("\nTesting with file Test7.csv\n")
    df = pd.read_csv("Test7.csv")
    print("Dataset entropy:", np.isclose(get_entropy_of_dataset(df),1.438862875041894))
    print("caprice Entropy:", np.isclose(get_entropy_of_attribute(df, 'caprice'),1.2921917219486119))
    print("caprice IG:", np.isclose(get_information_gain(df, 'caprice'),0.14667115309328205))
    print("topic Entropy:", np.isclose(get_entropy_of_attribute(df, 'topic'),1.0883213898497568))
    print("topic IG:", np.isclose(get_information_gain(df, 'topic'),0.3505414851921371))
    print("lmt Entropy:", np.isclose(get_entropy_of_attribute(df, 'lmt'),1.4134166106032882))
    print("lmt IG:", np.isclose(get_information_gain(df, 'lmt'),0.02544626443860576))
    print("lpss Entropy:", np.isclose(get_entropy_of_attribute(df, 'lpss'),1.3920622169341164))
    print("lpss IG:", np.isclose(get_information_gain(df, 'lpss'),0.04680065810777756))
    print("pb Entropy:", np.isclose(get_entropy_of_attribute(df, 'pb'),1.3591269643656734))
    print("pb IG:", np.isclose(get_information_gain(df, 'pb'),0.07973591067622054))
    
if __name__=="__main__":
	test_case()
