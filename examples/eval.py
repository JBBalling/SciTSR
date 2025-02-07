import json

from scitsr.eval import json2Relations, eval_relations, json2Table

PATH_TO_JSON = "" # put the path to your json file here

def example():
  json_path = PATH_TO_JSON
  with open(json_path) as fp: json_obj = json.load(fp)
  ground_truth_relations = json2Relations(json_obj, splitted_content=True)
  # your_relations should be a List of Relation.
  # Here we directly use the ground truth relations as the results.
  your_relations = ground_truth_relations
  precision, recall = eval_relations(
    gt=[ground_truth_relations], res=[your_relations], cmp_blank=True)
  f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  print("P: %.2f, R: %.2f, F1: %.2f" % (precision, recall, f1))
  # x = json2Table(json_obj, splitted_content=True)
  #print(json2Table(json_obj, splitted_content=True).cells)
  #print(json2Table(json_obj, splitted_content=True).__getitem__((1,1)))


if __name__ == "__main__":
  example()  