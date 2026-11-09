import unittest
import tempfile
from pathlib import Path
from spam_classifier import nb_train, nb_test, f_score, load_data

class ClassifierTests(unittest.TestCase):
    def setUp(self): self.model=nb_train(['meeting project','meeting team','prize win','prize claim'],[0,0,1,1])
    def test_counts(self):
        self.assertEqual(self.model['ham_count'],2);self.assertEqual(self.model['spam_fd']['prize'],2)
    def test_scoring_modes(self):
        for logs in [False,True]:
            for smooth in [False,True]:
                self.assertEqual(nb_test(['MEETING','win prize'],self.model,logs,smooth),[0,1])
    def test_priors_and_unknown(self):
        model=nb_train(['meeting','team','work','prize'],[0,0,0,1])
        for logs in [False,True]: self.assertEqual(nb_test(['','unseen'],model,logs,True),[0,0])
    def test_zero_probability(self):
        for logs in [False,True]: self.assertEqual(nb_test(['meeting prize'],self.model,logs,False),[0])
    def test_long_log_document(self): self.assertEqual(nb_test(['prize '*10000],self.model,True,True),[1])
    def test_unknown_invariance(self):
        self.assertEqual(nb_test(['meeting unknown'],self.model,True,True),nb_test(['meeting'],self.model,True,True))
    def test_invalid_training(self):
        for x,y in [([],[]),(['a'],[]),(['a'],[3]),(['a'],[0]),(['','b'],[0,1])]:
            with self.assertRaises(ValueError): nb_train(x,y)
    def test_f1(self):
        self.assertEqual(f_score([1,1,0,0],[1,0,1,0]),0.5)
        self.assertEqual(f_score([0],[0]),0)
    def test_load_data(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'HAM.1.txt').write_text('synthetic meeting',encoding='utf-8')
            (Path(d)/'SPAM.1.txt').write_text('synthetic prize',encoding='utf-8')
            self.assertEqual(load_data(d),(['synthetic meeting','synthetic prize'],[0,1]))

if __name__=='__main__': unittest.main()
