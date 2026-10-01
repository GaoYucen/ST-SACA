"""Pure-stdlib pilot contract checks; numerical validation runs separately in CI."""
import copy
import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'tools/engineering_training_pilot.py'
SPEC=importlib.util.spec_from_file_location('pilot_contract',FILE)
P=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P)


def dataset():
    rng=random.Random(123);rows=[];seen=set()
    for n in (5,6):
        for split,count in (('train',32),('validation',8)):
            for _ in range(count):
                while True:
                    ids=rng.sample(range(30),n);weights=[30]+[0]*(n-1)
                    identity=P.instance_id(ids,weights)
                    if identity not in seen: break
                seen.add(identity)
                coordinates=[[104.1+k*.001,30.7+k*.001] for k in ids]
                order=list(range(n))
                rows.append(dict(identity=identity,split=split,station_ids=ids,station_coords=coordinates,passengers=weights,local_optimal_order=order,optimal_passenger_km=P.passenger_cost(coordinates,weights,order)))
    return dict(protocol=P.PROTOCOL,station_sha256=P.STATION_SHA,seed=P.DATA_SEED,capacity=30,objective='sum_passenger_weighted_cumulative_haversine_km_no_return',instances=rows)


class PilotContractTests(unittest.TestCase):
    def test_weighted_open_route_units(self):
        with patch.object(P,'DEPOT',[0,0]):
            actual=P.passenger_cost([[1,0],[2,0]],[2,1],[0,1])
            expected=4*6371*math.pi/180
            self.assertAlmostEqual(actual,expected,places=8)

    def test_zero_weight_tie_allowed(self):
        with patch.object(P,'DEPOT',[0,0]):
            self.assertAlmostEqual(P.passenger_cost([[1,0],[2,0]],[0,30],[0,1]),P.passenger_cost([[1,0],[2,0]],[0,30],[1,0]),places=8)

    def test_local_route_permutation_required(self):
        with self.assertRaisesRegex(AssertionError,'route_permutation'):
            P.passenger_cost([[1,0],[2,0]],[2,1],[0,0])

    def test_identity_ignores_input_permutation(self):
        self.assertEqual(P.instance_id([1,8,4],[2,3,25]),P.instance_id([4,1,8],[25,2,3]))

    def test_valid_split(self):
        self.assertEqual(len(P.validate_dataset(dataset())),80)

    def test_cross_split_duplicate_rejected(self):
        d=dataset();d['instances'][32]=copy.deepcopy(d['instances'][0]);d['instances'][32]['split']='validation'
        with self.assertRaisesRegex(AssertionError,'split_leakage_or_duplicate'): P.validate_dataset(d)

    def test_capacity50_rejected(self):
        d=dataset();d['instances'][0]['passengers'][0]=50
        with self.assertRaisesRegex(AssertionError,'passenger_capacity'): P.validate_dataset(d)

    def test_wrong_label_cost_rejected(self):
        d=dataset();d['instances'][0]['optimal_passenger_km']+=1
        with self.assertRaisesRegex(AssertionError,'label_cost'): P.validate_dataset(d)

    def test_validation_count_drift_rejected(self):
        d=dataset();d['instances'][0]['split']='validation'
        with self.assertRaisesRegex(AssertionError,'split_counts'): P.validate_dataset(d)

    def test_station_coordinates_bound_to_ids(self):
        d=dataset();stations=[[104.1+k*.001,30.7+k*.001] for k in range(30)]
        P.validate_dataset(d,stations)
        d['instances'][0]['station_coords'][0][0]+=1
        with self.assertRaisesRegex(AssertionError,'station_coordinate_binding'): P.validate_dataset(d,stations)

    def test_required_artifact_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as root,patch.object(P,'git_head',return_value='a'*40):
            p=Path(root);(p/'other.json').write_text('{}')
            P.write_json(p/'pilot_manifest.json',dict(protocol=P.PROTOCOL,stage='labels',research_commit='a'*40,asset_role='engineering_pilot_only',scientific_result_verified=False,artifacts={'other.json':P.fingerprint(p/'other.json')}))
            with self.assertRaisesRegex(AssertionError,'artifact_manifest'): P.load_stage(p,'labels')

    def test_extra_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as root,patch.object(P,'git_head',return_value='a'*40):
            p=Path(root)
            P.write_json(p/'pilot_manifest.json',dict(protocol=P.PROTOCOL,stage='labels',research_commit='a'*40,asset_role='engineering_pilot_only',scientific_result_verified=False,artifacts={'pilot_labels.json':'0'*64,'extra.json':'0'*64}))
            with self.assertRaisesRegex(AssertionError,'artifact_manifest'): P.load_stage(p,'labels')

    def test_malformed_hash_rejected(self):
        with tempfile.TemporaryDirectory() as root,patch.object(P,'git_head',return_value='a'*40):
            p=Path(root)
            P.write_json(p/'pilot_manifest.json',dict(protocol=P.PROTOCOL,stage='labels',research_commit='a'*40,asset_role='engineering_pilot_only',scientific_result_verified=False,artifacts={'pilot_labels.json':'invalid'}))
            with self.assertRaisesRegex(AssertionError,'artifact_name_or_hash'): P.load_stage(p,'labels')

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'target';target.write_text('x');link=Path(root)/'link';link.symlink_to(target)
            with self.assertRaisesRegex(AssertionError,'symlink_refused'): P.fingerprint(link)

    def test_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'output.json';p.write_text('original')
            with self.assertRaises(FileExistsError): P.write_json(p,{'x':1})
            self.assertEqual(p.read_text(),'original')

    def test_cli_help(self):
        p=subprocess.run([sys.executable,'-B',str(FILE),'--help'],capture_output=True,timeout=5)
        self.assertEqual(p.returncode,0)

    def test_cli_seed_out_of_range(self):
        p=subprocess.run([sys.executable,'-B',str(FILE),'sac','--seed','2','--out','/tmp/unused-pilot-contract'],capture_output=True,timeout=5)
        self.assertEqual(p.returncode,2)

    def test_heavy_dependencies_not_imported_by_module(self):
        self.assertNotIn('torch',P.__dict__)
        self.assertNotIn('numpy',P.__dict__)


if __name__=='__main__': unittest.main()
