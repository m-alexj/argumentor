from art import significance_tests, scores, aggregators

test = significance_tests.ApproximateRandomizationTest(
    scores.Scores.from_file(open('/hits/fast/nlp/judeaax/event2/n/3/arguments_sentence.txt')), 
    scores.Scores.from_file(open('/data/nlp/judeaax/event.files/event.1/decoder.out/arguments_persentence.txt')), 
    aggregators.f_1)
print(test.run())