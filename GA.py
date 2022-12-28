import random
import operator
import copy
import matplotlib.pyplot as plt

def BinAndDec (value,type):
    #if values less than 14 bits hence make sure 14 bits are passed for and 14 for y
    lb = -5.000 #lower bound defined in question
    ub = +5.000 #upper bound defined in question
    p = 0.001   # precision
    if type == 'b2d':
        # means value is in binary (list of 1 and 0s) have to convert it into decimal
        binaryStringList = [str(i) for i in value]
        decimalValue = int("".join(binaryStringList),2)
        decimalValue -=1
        decimalValue=decimalValue*p
        decimalValue = decimalValue+lb
        return round(decimalValue,3)
    else:
        #means value is decimal have to make it binary and return binary value in the form of list
        val = int(((value - lb)/p)+1)
        binVal = bin(val).replace('0b','').zfill(14)
        result = [int(i) for i in binVal]
        return result

def fitnessFunction (x,y):
    z=((4-(2.1*pow(x,2))+(pow(x,4)/3))*pow(x,2))+(x*y)+((-4+(4*pow(y,2)))*pow(y,2))

    return z

def GeneticAlgorithm(population,totalGenerations):
    lb = -5.000
    up = 5.000
    currentGeneration = []
    listAverageFitness = []
    listBestFitness = []
    numberOfGen = []
    bestValue = None
    for i in range(population):
        individual_x = round(random.uniform(-5.000,5.000), 3)
        individual_y = round(random.uniform(-5.000,5.000), 3)
        fitnessValue = fitnessFunction(individual_x,individual_y)
        binaryX = BinAndDec(individual_x,'d2b')
        binaryY = BinAndDec(individual_y,'d2b')
        currentGeneration.append({
            'xValue':individual_x,
            'yValue':individual_y,
            'fitnessScore':fitnessValue,
            'x_binary':binaryX,
            'y_binary':binaryY,
            'chromosome':[binaryX,binaryY]
        })
    iter = 0
    while(iter<totalGenerations):
        numberOfGen.append(iter)
        tempBest= 100000000000
        tempAverage = 0
        for ind in currentGeneration:
            tempAverage+=ind['fitnessScore']
            if ind['fitnessScore']<tempBest:
                tempBest=ind['fitnessScore']
        tempAverage=tempAverage/len(currentGeneration)
        listAverageFitness.append(tempAverage)
        listBestFitness.append(tempBest)
        #parent selection
        chosenParents = selection_parents(currentGeneration,population)
        offspringsResults = func_crossover(chosenParents)
        survivedChildren = mutation(offspringsResults,population)
        if bestValue == None:
            bestValue=survivedChildren[0]
        else:
            if survivedChildren[0]['fitnessScore']<bestValue['fitnessScore']:
                bestValue = survivedChildren[0]
        currentGeneration = survivedChildren
        iter+=1
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.suptitle("Graphs")
    ax1.plot(numberOfGen,listBestFitness)
    ax1.set_title("Best Fitness Vs Number of Generations")
    ax2.plot(numberOfGen,listAverageFitness)
    ax2.set_title("Average Fitness Vs Number of Generations")
    plt.show()
    return bestValue


def selection_parents(presentGen,populationSize):
    # implementing tournament selection
    selectedParents = []
    iter = 0
    samplingK = int(0.05*populationSize)
    while(iter<populationSize):
        min_fitness = 100000000
        min_fitness_parent = None
        sampleParents = random.sample(presentGen,samplingK)
        for i in range(len(sampleParents)):
            if sampleParents[i]['fitnessScore']<min_fitness:
                min_fitness = sampleParents[i]['fitnessScore']
                min_fitness_parent = i
            else:
                continue
        selectedParents.append(sampleParents[min_fitness_parent])
        iter+=1
    return selectedParents

def func_crossover(allParentsList):
    parentsList = copy.deepcopy(allParentsList)
    offspringList = []
    for i in range(len(parentsList)):
        parent1 = parentsList[i]
        if i+1>=len(parentsList):
            parent2 = parentsList[-1]
        else:
            parent2 = parentsList[i+1]
        #Now we have parent 1 and parent 2
        # As we are assuming that the probability of crossover
        # is high we apply 1 point crossover
        # interchanging y values for 2 parents will 2 new offsprings
        offspringOne = {}
        offspringTwo = {}
        for key in parent1.keys():
            if key=='xValue' or key=='x_binary':
                offspringOne[key] = parent1[key]
                offspringTwo[key] = parent2[key]
            elif key == 'yValue' or key == 'y_binary':
                offspringOne[key] = parent2[key]
                offspringTwo[key] = parent1[key]
            elif key == 'fitnessScore':
                fitnessValueO1 = fitnessFunction(offspringOne['xValue'],offspringOne['yValue'])
                offspringOne[key] = fitnessValueO1
                fitnessValueO2 = fitnessFunction(offspringTwo['xValue'],offspringTwo['yValue'])
                offspringTwo[key] = fitnessValueO2
            else:
                #only 'chromosome' left
                offspringOne[key] = [offspringOne['x_binary'],offspringOne['y_binary']]
                offspringTwo[key] = [offspringTwo['x_binary'],offspringTwo['y_binary']]
        offspringList.append(offspringOne)
        offspringList.append(offspringTwo)

    return offspringList


def mutation(allOffsprings,populationSize):
    firstbound = 1/28      # as chromosome length remains same
    secondbound = 1/populationSize #as population
    pm = (firstbound+secondbound)/4
    for child in allOffsprings:
        for i in range(28):     #as the chromosome length is 28
            pTestValue = random.random()
            if pTestValue<=pm:
                if i<14:
                    child['chromosome'][0][i] = int(not bool(child['chromosome'][0][i]))
                else:
                    child['chromosome'][1][i-14] = int(not bool(child['chromosome'][1][i-14]))
            else:
                continue
        
        binaryValX = child['chromosome'][0]
        binaryValY = child['chromosome'][1]
        integerValX = BinAndDec(binaryValX,'b2d')
        integerValy = BinAndDec(binaryValY,'b2d')
        childFitness = fitnessFunction(integerValX,integerValy)
        child['fitnessScore'] = childFitness
        child['xValue'] = integerValX
        child['yValue'] = integerValy
        child['x_binary'] = binaryValX
        child['y_binary'] = binaryValY

    sortedChildren = sorted(allOffsprings, key=lambda x: x['fitnessScore'])
    requiredPopulation = int(len(sortedChildren)/2)
    finalChildrenForPopulation = sortedChildren[:requiredPopulation]
    return finalChildrenForPopulation


def main():
    population = 1000
    totalGenerations = 125
    finalResult = GeneticAlgorithm(population,totalGenerations)
    print(finalResult)


if __name__ == '__main__':
    main()