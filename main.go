// KNN project main.go
// Using data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

type irisRecord struct {
	sepalLength float64
	sepalWidth  float64
	petalLength float64
	petalWidth  float64
	species     string
}

func main() {
	irisData, err := os.Open("iris.data")
	errHandle(err)
	defer irisData.Close()

	reader := csv.NewReader(irisData)
	reader.Comma = ','

	var recordSet []irisRecord

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		errHandle(err)

		recordSet = append(recordSet, parseIrisRecord(record))
	}

	var testSet []irisRecord
	var trainSet []irisRecord

	for i := range recordSet {
		if rand.Float64() < 0.4 {
			trainSet = append(trainSet, recordSet[i])
		} else {
			testSet = append(testSet, recordSet[i])
		}
	}

	var predictions []string
	k := 3

	for x := range testSet {
		neighbors := getNeighbors(trainSet, testSet[x], k)
		result := getResponse(neighbors)
		predictions = append(predictions, result[0].key)
		fmt.Printf("Predicted: %s, Actual: %s\n", result[0].key, testSet[x].species)
	}

	accuracy := getAccuracy(testSet, predictions)
	fmt.Printf("Accuracy: %f%s\n", accuracy, "%")
}

func getAccuracy(testSet []irisRecord, predictions []string) float64 {
	correct := 0

	for x := range testSet {
		if testSet[x].species == predictions[x] {
			correct += 1
		}
	}

	return (float64(correct) / float64(len(testSet))) * 100.00
}

type classVote struct {
	key   string
	value int
}

type sortedClassVotes []classVote

func (scv sortedClassVotes) Len() int           { return len(scv) }
func (scv sortedClassVotes) Less(i, j int) bool { return scv[i].value < scv[j].value }
func (scv sortedClassVotes) Swap(i, j int)      { scv[i], scv[j] = scv[j], scv[i] }

func getResponse(neighbors []irisRecord) sortedClassVotes {
	classVotes := make(map[string]int)

	for x := range neighbors {
		response := neighbors[x].species
		if contains(classVotes, response) {
			classVotes[response] += 1
		} else {
			classVotes[response] = 1
		}
	}

	scv := make(sortedClassVotes, len(classVotes))
	i := 0
	for k, v := range classVotes {
		scv[i] = classVote{k, v}
		i++
	}

	sort.Sort(sort.Reverse(scv))
	return scv
}

type distancePair struct {
	record   irisRecord
	distance float64
}

type distancePairs []distancePair

func (slice distancePairs) Len() int           { return len(slice) }
func (slice distancePairs) Less(i, j int) bool { return slice[i].distance < slice[j].distance }
func (slice distancePairs) Swap(i, j int)      { slice[i], slice[j] = slice[j], slice[i] }

func getNeighbors(trainingSet []irisRecord, testRecord irisRecord, k int) []irisRecord {
	var distances distancePairs
	for i := range trainingSet {
		dist := euclidianDistance(testRecord, trainingSet[i])
		distances = append(distances, distancePair{trainingSet[i], dist})
	}

	sort.Sort(distances)

	var neighbors []irisRecord

	for x := 0; x < k; x++ {
		neighbors = append(neighbors, distances[x].record)
	}

	return neighbors
}

func euclidianDistance(instanceOne irisRecord, instanceTwo irisRecord) float64 {
	var distance float64

	distance += math.Pow((instanceOne.petalLength - instanceTwo.petalLength), 2)
	distance += math.Pow((instanceOne.petalWidth - instanceTwo.petalWidth), 2)
	distance += math.Pow((instanceOne.sepalLength - instanceTwo.sepalLength), 2)
	distance += math.Pow((instanceOne.sepalWidth - instanceTwo.sepalWidth), 2)

	return math.Sqrt(distance)
}

func parseIrisRecord(record []string) irisRecord {
	var iris irisRecord

	iris.sepalLength, _ = strconv.ParseFloat(record[0], 64)
	iris.sepalWidth, _ = strconv.ParseFloat(record[1], 64)
	iris.petalLength, _ = strconv.ParseFloat(record[2], 64)
	iris.petalWidth, _ = strconv.ParseFloat(record[3], 64)
	iris.species = record[4]

	return iris
}

func errHandle(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func contains(votesMap map[string]int, name string) bool {
	for s, _ := range votesMap {
		if s == name {
			return true
		}
	}

	return false
}
