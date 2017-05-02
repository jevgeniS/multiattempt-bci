class ContingencyTableBuilder(object):
    def build(self, prediction_result):
        tp = 0
        fn = 0
        fp = 0
        tn = 0

        for cls in prediction_result:
            # Positive
            predicted_classes=prediction_result[cls]
            if cls == "Relax":
                tp = len([i for i,x in enumerate(predicted_classes) if x == cls])
                fn = len([i for i,x in enumerate(predicted_classes) if x != cls])
            # Negative
            else:
                tn = len([i for i,x in enumerate(predicted_classes) if x == cls])
                fp = len([i for i,x in enumerate(predicted_classes) if x != cls])
        print "TP: {}, FN: {}, FP: {}, TN: {}".format(tp, fn, fp, tn)
