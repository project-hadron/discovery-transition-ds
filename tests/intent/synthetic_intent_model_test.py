import unittest
import os
import shutil
from pprint import pprint

import pandas as pd
from ds_discovery import SyntheticBuilder, Wrangle
from aistac.properties.property_manager import PropertyManager

from ds_discovery.intent.synthetic_intent import SyntheticIntentModel
from ds_discovery.intent.wrangle_intent import WrangleIntentModel


class SyntheticIntentModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # clean out any old environments
        for key in os.environ.keys():
            if key.startswith("HADRON"):
                del os.environ[key]

        os.environ["HADRON_PM_PATH"] = os.path.join("work", "config")
        os.environ["HADRON_DEFAULT_PATH"] = os.path.join("work", "data")
        try:
            os.makedirs(os.environ["HADRON_PM_PATH"])
            os.makedirs(os.environ["HADRON_DEFAULT_PATH"])
        except:
            pass
        PropertyManager._remove_all()

    def tearDown(self):
        try:
            shutil.rmtree("work")
        except:
            pass

    def test_model_columns_headers(self):
        builder = SyntheticBuilder.from_env(
            "test", default_save=False, default_save_intent=False, has_contract=False
        )
        tools: SyntheticIntentModel = builder.tools
        builder.set_source_uri(
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        )
        df = pd.DataFrame(index=range(300))
        result = tools.model_concat(
            df,
            other=builder.CONNECTOR_SOURCE,
            as_rows=False,
            headers=["survived", "sex", "fare"],
        )
        self.assertCountEqual(["survived", "sex", "fare"], list(result.columns))
        self.assertEqual(300, result.shape[0])

    def test_model_modifier(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5], "B": [1, 2, 3, 4, 5]})
        other = {"headers": ["A", "B"], "target": [2, 0.2]}
        result = tools.model_modifier(df, other)
        self.assertEqual([3.0, 4.0, 5.0, 6.0, 7.0], result["A"].to_list())
        self.assertEqual([1.2, 2.2, 3.2, 4.2, 5.2], result["B"].to_list())
        result = tools.model_modifier(df, other, modifier="add")
        self.assertEqual([3.0, 4.0, 5.0, 6.0, 7.0], result["A"].to_list())
        self.assertEqual([1.2, 2.2, 3.2, 4.2, 5.2], result["B"].to_list())
        result = tools.model_modifier(df, other, modifier="mul")
        self.assertEqual([2.0, 4.0, 6.0, 8.0, 10.0], result["A"].to_list())
        self.assertEqual([0.2, 0.4, 0.6, 0.8, 1.0], result["B"].to_list())
        result = tools.model_modifier(df, other, targets_header='headers', values_header='target', modifier='div')
        self.assertEqual([0.5, 1.0, 1.5, 2.0, 2.5], result["A"].to_list())
        self.assertEqual([5.0, 10.0, 15.0, 20.0, 25.0], result["B"].to_list())

    def test_model_merge(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5], "B": list("ABCDE")})
        other = {"A": [5, 2, 3, 1, 4], "X": list("VWXYZ"), "Y": list("VWXYZ")}
        # using left_on and right_on
        result = tools.model_merge(canonical=df, other=other, right_on="A", left_on="A")
        self.assertEqual((5, 4), result.shape)
        self.assertEqual(["A", "B", "X", "Y"], result.columns.to_list())
        # using on
        result = tools.model_merge(canonical=df, other=other, on="A")
        self.assertEqual((5, 4), result.shape)
        self.assertEqual(["A", "B", "X", "Y"], result.columns.to_list())
        # filter headers
        result = tools.model_merge(canonical=df, other=other, on="A", headers=["X"])
        self.assertEqual((5, 3), result.shape)
        self.assertEqual(["A", "B", "X"], result.columns.to_list())

    def test_model_merge_str(self):
        sb = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = sb.tools
        #build files
        sb.add_connector_persist('source', uri_file='sourcefile.parquet')
        sb.add_connector_persist('other', uri_file='otherfile.parquet')
        dfA = pd.DataFrame()
        dfA['ref'] = tools.get_number(10_000, 99_000, precision=0, at_most=1, size=1000)
        dfA['A'] = tools.get_category(selection=[1, 0], relative_freq=[9.1], size=1000)
        dfB = pd.DataFrame()
        dfB['ref'] = dfA['ref'].sample(frac=1)
        dfB['B'] = tools.get_category(selection=[1, 0], relative_freq=[8, 1], size=1000)
        sb.save_canonical('source', dfA)
        sb.save_canonical('other', dfB)
        # merge files
        wr = Wrangle.from_memory()
        wr.set_source('sourcefile.parquet')
        sb.add_connector_persist('other', uri_file='otherfile.parquet')
        df = wr.load_source_canonical()
        df = sb.tools.model_merge(df, other='other', on='ref')
        self.assertEqual((1000, 3), df.shape)
        self.assertEqual(['ref', 'A', 'B'], df.columns.to_list())


    def test_model_code(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5, 6], "B": ['M', 'F', 'F', 'U', 'U', 'F']})
        result = tools.model_encode_integer(df, headers='B')
        self.assertEqual(['A', 'B'], df.columns.to_list())
        self.assertEqual([2, 1, 1, 3, 3, 1], result.B.to_list())

    def test_modal_merge_nulls(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5], "B": list("ABCDE")})
        other = {"A": [5, 7, 3, 9, 4], "X": list("VWXYZ"), "Y": [9, 7, 12, 10, 2]}
        # using left_on and right_on
        result = tools.model_merge(canonical=df, other=other, how='left', right_on="A", left_on="A", replace_nulls=True)
        self.assertEqual((5, 4), result.shape)
        control = {'X': {0: '', 1: ''}, 'Y': {0: 0.0, 1: 0.0}}
        self.assertDictEqual(control, result.loc[:1,['X', 'Y']].to_dict())

    def test_missing_cca(self):
        # load titanic subset
        url = "https://raw.github.com/mattdelhey/kaggle-titanic/master/Data/train.csv"
        df = pd.read_csv(url)
        df = df[['survived', 'age']]
        # add gender with nulls
        builder = SyntheticBuilder.from_memory()
        df['gender'] = builder.tools.get_category(selection=['M', 'F', None], relative_freq=[0.5, 0.3, 0.1], size=df.shape[0], seed=31)
        # test
        wr = Wrangle.from_memory()
        tools: WrangleIntentModel = wr.tools
        result = tools.model_missing_cca(df)
        self.assertEqual(df.shape[0] - result.shape[0], 259)
        result = tools.model_missing_cca(df, threshold=0.15)
        self.assertEqual(df.shape[0] - result.shape[0], 101)

    def test_remove_unwanted_headers(self):
        builder = SyntheticBuilder.from_env(
            "test", default_save=False, default_save_intent=False, has_contract=False
        )
        builder.set_source_uri(
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        )
        selection = [builder.tools.select2dict(column="survived", condition="@==1")]
        result = builder.tools.frame_selection(
            canonical=builder.CONNECTOR_SOURCE,
            selection=selection,
            headers=["survived", "sex", "fare"],
        )
        self.assertCountEqual(["survived", "sex", "fare"], list(result.columns))
        self.assertEqual(1, result["survived"].min())
        result = builder.tools.frame_selection(
            canonical=builder.CONNECTOR_SOURCE, headers=["survived", "sex", "fare"]
        )
        self.assertEqual((891, 3), result.shape)

    def test_remove_unwanted_rows(self):
        builder = SyntheticBuilder.from_env(
            "test", default_save=False, default_save_intent=False, has_contract=False
        )
        builder.set_source_uri(
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        )
        selection = [builder.tools.select2dict(column="survived", condition="@==1")]
        result = builder.tools.frame_selection(
            canonical=builder.CONNECTOR_SOURCE, selection=selection
        )
        self.assertEqual(1, result["survived"].min())

    def test_model_sample_map(self):
        builder = SyntheticBuilder.from_memory(default_save_intent=False)
        result = builder.tools.model_sample_map(
            pd.DataFrame(), sample_map="us_healthcare_practitioner"
        )
        self.assertEqual((192865, 6), result.shape)
        result = builder.tools.model_sample_map(
            pd.DataFrame(index=range(50)), sample_map="us_healthcare_practitioner"
        )
        self.assertEqual((50, 6), result.shape)
        result = builder.tools.model_sample_map(
            pd.DataFrame(index=range(50)),
            sample_map="us_healthcare_practitioner",
            headers=["pcp_tax_id"],
        )
        self.assertEqual((50, 1), result.shape)

    def test_model_us_person(self):
        builder = SyntheticBuilder.from_memory(default_save_intent=False)
        df = pd.DataFrame(index=range(300))
        result = builder.tools.model_sample_map(canonical=df, sample_map="us_persona")
        self.assertCountEqual(
            ["first_name", "middle_name", "gender", "family_name", "email"],
            result.columns.to_list(),
        )
        self.assertEqual(300, result.shape[0])
        df = pd.DataFrame(index=range(1000))
        df = builder.tools.model_sample_map(
            canonical=df, sample_map="us_persona", female_bias=0.3
        )
        self.assertEqual((1000, 5), df.shape)
        print(df["gender"].value_counts().loc["F"])

    def test_model_inc(self):
        df = pd.DataFrame(data={"A": [1, 2, 3, 4, 5], "B": [1, 2, 3, 4, 5], "C": [0, 0, 0, 0, 0]})
        other = pd.DataFrame(data={"A": [1, 7, 5, 3], "B": [1, 2, 1.5, 0], "C": [1, 0, 1, 0]})
        other = other.loc[other.loc[:,'A'].isin(df.loc[:,'A']), :].index.to_list()
        # result = df.combine(other, lambda s1, s2: s2 + s1 if len(s2.mode()) else s1)
        print(other)

    def test_model_concat(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri(
            "titanic",
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        )
        df = pd.DataFrame(index=range(1973))
        df = tools.model_concat(df, other="titanic")
        self.assertEqual((1973, 15), df.shape)
        df = pd.DataFrame(index=range(100))
        df = tools.model_concat(
            df, other="titanic", headers=["class", "embark_town", "survived", "sex"]
        )
        self.assertEqual((100, 4), df.shape)

    def test_model_group(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_uri(
            "titanic",
            uri="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        )
        df = tools.model_group(
            "titanic", headers="fare", group_by=["survived", "sex"], aggregator="sum"
        )
        self.assertEqual((4, 3), df.shape)
        df = tools.model_group(
            "titanic",
            headers=["class", "embark_town"],
            group_by=["survived", "sex"],
            aggregator="set",
            list_choice=2,
        )
        # print(df.loc[:, ['class', 'embark_town']])
        self.assertEqual((4, 4), df.shape)
        self.assertCountEqual(
            ["class", "embark_town", "survived", "sex"], df.columns.to_list()
        )
        df = tools.model_group(
            "titanic",
            headers=["fare", "survived"],
            group_by="sex",
            aggregator="sum",
            include_weighting=True,
        )
        self.assertEqual((2, 4), df.shape)
        self.assertCountEqual(
            ["survived", "sex", "fare", "weighting"], df.columns.to_list()
        )

    def test_model_explode(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [[2, 2], [3], [7, 8, 9]]})
        wr = Wrangle.from_memory(default_save_intent=False)
        df = wr.tools.model_explode(df, header="B")
        self.assertEqual([1, 1, 2, 3, 3, 3], df["A"].to_list())
        self.assertEqual([2, 2, 3, 7, 8, 9], df["B"].to_list())

    def test_model_sample(self):
        builder = SyntheticBuilder.from_env(
            "test", default_save=False, default_save_intent=False, has_contract=False
        )
        tools: SyntheticIntentModel = builder.tools
        builder.add_connector_persist(
            connector_name="sample", uri_file="sample.parquet"
        )
        sample = pd.DataFrame()
        sample["age"] = [20, 34, 50, 75]
        sample["gender"] = list("MMFM")
        builder.persist_canonical(connector_name="sample", canonical=sample)
        df = pd.DataFrame(index=range(1973))
        df = tools.model_sample(df, other="sample", headers=["age", "gender"])
        self.assertEqual((1973, 2), df.shape)

    def test_model_dict(self):
        builder = SyntheticBuilder.from_memory()
        tools: SyntheticIntentModel = builder.tools
        sample = [
            {"task": "members_sim", "source": 100000},
            {"task": "pcp_sim", "source": 0},
            {"task": "members_gen", "source": "members_sim", "persist": True},
            None,
        ]
        df = pd.DataFrame(data={"A": [5, 2, 3, 4], "X": sample, "Y": list("VWXY")})
        df = tools._model_dict_column(df, header="X")
        self.assertCountEqual(
            ["A", "Y", "task", "source", "persist"], df.columns.to_list()
        )
        # as strings
        sample = [
            "{'task': 'members_sim', 'source': 100000}",
            "{'task': 'pcp_sim', 'source': 0}",
            "{'task': 'members_gen', 'source': 'members_sim', 'persist': True}",
            None,
        ]
        df = pd.DataFrame(data={"A": [5, 2, None, 4], "X": sample, "Y": list("VWXY")})
        df = tools._model_dict_column(df, header="X", convert_str=True)
        self.assertCountEqual(
            ["A", "Y", "task", "source", "persist"], df.columns.to_list()
        )
        # replace nulls
        df = pd.DataFrame(data={"A": [5, 2, None, 4], "X": sample, "Y": list("VWXY")})
        df = tools._model_dict_column(
            df, header="X", convert_str=True, replace_null="default"
        )
        self.assertEqual(
            ["default", "default", "default"],
            df.loc[3, ["task", "source", "persist"]].to_list(),
        )

    def test_model_analysis(self):
        uri = (
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        )
        builder = SyntheticBuilder.from_memory()
        builder.add_connector_uri(connector_name="titanic", uri=uri)
        tools: SyntheticIntentModel = builder.tools
        sample = builder.load_canonical("titanic")
        df = tools.model_analysis(1000, other="titanic")
        self.assertEqual((1000, 15), df.shape)
        self.assertCountEqual(sample.columns.to_list(), df.columns.to_list())

    def test_raise(self):
        with self.assertRaises(KeyError) as context:
            env = os.environ["NoEnvValueTest"]
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
