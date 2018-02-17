# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8

"""Read text files and load embeddings."""
from __future__ import absolute_import
from __future__ import print_function

UNKNOWN_IDX = 0

APACHE_REPO_URL = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'

GLOVE_PRETRAINED_FILE_SHA1 = \
    {'glove.42B.300d.zip': 'f8e722b39578f776927465b71b231bae2ae8776a',
     'glove.6B.zip': 'b64e54f1877d2f735bdd000c1d7d771e25c7dfdc',
     'glove.840B.300d.zip': '8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d',
     'glove.twitter.27B.zip': 'dce69c404025a8312c323197347695e81fd529fc'}

GLOVE_PRETRAINED_ARCHIVE_SHA1 = \
    {'glove.42B.300d.txt': '876767977d6bd4d947c0f84d44510677bc94612a',
     'glove.6B.50d.txt': '21bf566a9d27f84d253e0cd4d4be9dcc07976a6d',
     'glove.6B.100d.txt': '16b1dbfaf35476790bd9df40c83e2dfbd05312f1',
     'glove.6B.200d.txt': '17d0355ddaa253e298ede39877d1be70f99d9148',
     'glove.6B.300d.txt': '646443dd885090927f8215ecf7a677e9f703858d',
     'glove.840B.300d.txt': '294b9f37fa64cce31f9ebb409c266fc379527708',
     'glove.twitter.27B.25d.txt':
         '767d80889d8c8a22ae7cd25e09d0650a6ff0a502',
     'glove.twitter.27B.50d.txt':
         '9585f4be97e286339bf0112d0d3aa7c15a3e864d',
     'glove.twitter.27B.100d.txt':
         '1bbeab8323c72332bd46ada0fc3c99f2faaa8ca8',
     'glove.twitter.27B.200d.txt':
         '7921c77a53aa5977b1d9ce3a7c4430cbd9d1207a'}

FAST_TEXT_FILE_SHA1 = \
    {'wiki.ab.vec': '9d89a403a9a866d3da8dd8cfab849f59ee499343',
     'wiki.ace.vec': '85d00074f7a08626f39da6a0c8a5cfa250096ab9',
     'wiki.ady.vec': '9d17d74f0348224cdebf8a831e61af0825f8952d',
     'wiki.aa.vec': '5cce30fc85471572c498f278bbe495184577363e',
     'wiki.af.vec': '999e64bcd8dab8de42cb1feceeca360def35324d',
     'wiki.ak.vec': '6092b8af335c2dc93e8df2bbf1d715f01e637bb4',
     'wiki.sq.vec': 'd07ffed553f5eb4756d0a1548a7ba9a51a52f7c6',
     'wiki.als.vec': '96052e96870695cca50857b5fde5f9f42219139a',
     'wiki.am.vec': 'dff7fcdd8f5ba0638ab9e1758a89800766156d72',
     'wiki.ang.vec': 'a7c30e02422d97d23a0701279c5c1c03159130a5',
     'wiki.ar.vec': 'c46e2142f799cc385bd25f0c0a8943ca565505a4',
     'wiki.an.vec': '5b4c2b1de5c04e4e0be83841410ca84c47305d21',
     'wiki.arc.vec': 'fd3ad743103f80cde9cfc048d7ca509e50efb35a',
     'wiki.hy.vec': '21f9259d04cfd22db446a45d3622af225f00cf20',
     'wiki.roa_rup.vec': 'e31a44353cd84b976586c8df35a2ab58318120f0',
     'wiki.as.vec': 'cad5883b5147cbe6cdbf604f65cabdb675a59258',
     'wiki.ast.vec': '89a90357101953b7c292697fd050c00fe5c38ac5',
     'wiki.av.vec': '99976a63ca8c4231f808fd4314f0433db35e290d',
     'wiki.ay.vec': 'be359dad25b2c742d3abfa94c5f5db13f86c730e',
     'wiki.az.vec': '9581d55d9056ad398a153c37b502f3a07867d091',
     'wiki.bm.vec': 'f36a19c95e90865f6518d4487e59f363b47bd865',
     'wiki.bjn.vec': '5f134cf288e8042dcd048a3ee76159aab42c7288',
     'wiki.map_bms.vec': 'e7deab5fdd38fa3331b1bcb4a16432b38c512e21',
     'wiki.ba.vec': '22147ee16b2d163cc88d09a035264fd0c10dab68',
     'wiki.eu.vec': '5e72f4ef93666971fea5d2180b354e0a0821ba91',
     'wiki.bar.vec': '96130f1f2e5bffdd06c202ad4472e5234020980a',
     'wiki.be.vec': '6cf81322cd7b046a7f02ec4c4960ad27045383fa',
     'wiki.bn.vec': '6fc3bfd9af455719f55bee0bea31b11afc70cf06',
     'wiki.bh.vec': 'ab2d29017afa015c49566a6d9bf75393c23ac4c0',
     'wiki.bpy.vec': 'c2bb15487c4bdb8fa869772694300ae1fee73896',
     'wiki.bi.vec': '15785220cd6e6c86cc87e7d3f3322a5541a4fe5d',
     'wiki.bs.vec': 'c4943a290819ceae1611dd11179b40aab0df0471',
     'wiki.br.vec': 'df44e16abd2017e2a1b6c6588ee02779b19907f6',
     'wiki.bug.vec': '942d8f7dadde5faa33aa72862501434f48e29f60',
     'wiki.bg.vec': '7c1cc6d0c52b038e4b7173259b0c009f242cf486',
     'wiki.my.vec': 'e7c7989e32b23ca1a9caf534cc65ecaf9e1b9112',
     'wiki.bxr.vec': 'eaf767690c6b194605ae778719212e3874873d4c',
     'wiki.zh_yue.vec': 'd2ac1ab9eb1a908797644f83f259c90cb3c1a350',
     'wiki.ca.vec': 'f5971edee11c939f6a7accfd33a9a45caa54141a',
     'wiki.ceb.vec': 'b8516a55537b8f80c927d77d95cdf7e4ff849a05',
     'wiki.bcl.vec': 'd4117b5c443438ddfa608b10a5be2c2501817e7e',
     'wiki.ch.vec': '46803f3a1734f6a7b0d8cb053bbb86a6915d02e9',
     'wiki.cbk_zam.vec': '6fef47b4559eec402ce371de20dfb018acd6347d',
     'wiki.ce.vec': '1d94b0168a773895b23889f7f07d7cf56c11a360',
     'wiki.chr.vec': '8501bf86b41074ed6c8d15b9209ef7ce83122e70',
     'wiki.chy.vec': '26c87688551ffe3a0c7a5952e894306651e62131',
     'wiki.ny.vec': '4e066fe113630fdfbcaf8844cc4ad64388db98d0',
     'wiki.zh.vec': '117ab34faa80e381641fbabf3a24bc8cfba44050',
     'wiki.cho.vec': 'cec6778f025fa9ae4134046c6c3a6291bd9c63f9',
     'wiki.cv.vec': '9cdb0bee5a0fea030def85597dba7108f21b0424',
     'wiki.zh_classical.vec': '840981c83dd8e5cb02d1cd695e2fe0870941316c',
     'wiki.kw.vec': 'f9eaa35a7e4f077f6de85c7801f74582f91b52c1',
     'wiki.co.vec': 'af876a918594e5541207bc12f17bfc4268df7b93',
     'wiki.cr.vec': '61dd9f044b7dfa56dcf1c3c07c7504c569420528',
     'wiki.crh.vec': 'c0d2310a1207fcacc94b25b149420b33bf835015',
     'wiki.hr.vec': '0c96f9af092cf8a84b03aec1426cd23921671489',
     'wiki.cs.vec': 'f3ec1502aeee6a550d8cf784273fa62f61419a4e',
     'wiki.da.vec': '526947dab1ffbc1465c7a766f2bca4de50676b08',
     'wiki.dv.vec': 'e135ba97c711a021bc3317db2b95db5212c17658',
     'wiki.nl.vec': 'd796ee27e37b7d1d464e03c265c31ab62b52533e',
     'wiki.nds_nl.vec': '1cd96d12e78e5cd3f65ca2773a17696bda387b9f',
     'wiki.dz.vec': '4cc1c6cf4aa4676d40a3145d5d4623569e430f89',
     'wiki.pa.vec': '4939d0db77a5b28d7d5aab0fab4f999d93b2053e',
     'wiki.arz.vec': '5e904087043b91f4945dd708f4230fdf51360132',
     'wiki.eml.vec': 'de6be7a2ffdda226eec730dd54b4c614bd7f5dca',
     'wiki.en.vec': 'c1e418f144ceb332b4328d27addf508731fa87df',
     'wiki.myv.vec': '7de0927fd3d65677de7f770b3bd57c73b58df85d',
     'wiki.eo.vec': 'b56998fd69f66755b722a9481a9bdaf10f62c9aa',
     'wiki.et.vec': '64d56b66c02d5e49b1b66a85854d67d2dd9ebd41',
     'wiki.ee.vec': 'f2212f58ec082321bc9b93873cd22702d0a64d64',
     'wiki.ext.vec': '456c5632b13a0f136cd180ebe2dda67b83f78397',
     'wiki.fo.vec': 'eead8ddc7bb74b12b16784723abf802bb51f844d',
     'wiki.hif.vec': '49697cf784814d3f1a47559724028e0fc0940d36',
     'wiki.fj.vec': 'c70fca34a7e43143600c54d7bf199b88846ac6f2',
     'wiki.fi.vec': '91d19baae994d7e556b5b5938be2dc6013f9c706',
     'wiki.frp.vec': '0eb70a613ccf807c7308c1f62535f0606465029d',
     'wiki.fr.vec': 'b092229005a65d8683a4112852fe6eb8161a6917',
     'wiki.fur.vec': 'd4a595cffa1abcdcf4229ba15277179ce5d20bc6',
     'wiki.ff.vec': '57ea8febb24ba8ac4421ec97ed8918d44c69f42c',
     'wiki.gag.vec': 'c82ec7a5d081f0673661824f4fc34345dee255f0',
     'wiki.gl.vec': '8888bb8f3d70b36729b9ae479fe3765e0c083862',
     'wiki.gan.vec': 'aeea01c2c4a7c44d6e8c31845560baf43d7afb9c',
     'wiki.ka.vec': '8b92b73f27f9b77818211e053a33985589de7c62',
     'wiki.de.vec': '2ed2696afe55f023b0040b238d9a47e5fedfe48b',
     'wiki.glk.vec': '20a7759075916e10531f5b3577302353cef565cd',
     'wiki.gom.vec': '5a1193d9e5d49d06354c14e2b7c01bea176e13f1',
     'wiki.got.vec': 'cc5aaf4c305f4f1f788b4829e644823f8495a23a',
     'wiki.el.vec': '6f034271390feaa6f9d7d16f933ddef637755979',
     'wiki.kl.vec': '390406cc33e02f86cfaf7ae273193679924f7413',
     'wiki.gn.vec': '98594af7897c5a1f35885ddecc77556a7e7ae981',
     'wiki.gu.vec': 'f9e13452eb63d92bea44c7c3db8fba9945c7000e',
     'wiki.ht.vec': '5039dfb58a074ac046813f2dae81159be8c5213f',
     'wiki.hak.vec': '9e83512d34c7f81739492bf0abbb25ff1ef88573',
     'wiki.ha.vec': '677a24efeeb1bcb8c0a931407775f18b18e875ae',
     'wiki.haw.vec': 'c23a50529dc010401c99833c8f990c1b27843db3',
     'wiki.he.vec': '55534560247394669e3f5c169136770c93bc2708',
     'wiki.hz.vec': '7605e06dd708920f73a80473816a8d684c116bd8',
     'wiki.mrj.vec': 'aa1c1ecba1ffd6b42c8d9659a8a04ab328ae1650',
     'wiki.hi.vec': '8049bb8604bc049d48bd934e27b0e184c480a413',
     'wiki.ho.vec': 'ef6b84d508d4d0a4c4cf90facaca1eebe62b2187',
     'wiki.hu.vec': 'cd777e9efca3d4bd97c89f01690cfa4840d9c46f',
     'wiki.is.vec': 'ae0b018f92b3e218f2dacb2045a8f0a0446788a5',
     'wiki.io.vec': 'af0c480c5872bff31d82e767c1116da2a6be0c00',
     'wiki.ig.vec': 'd2d1643b4fb1a18a4d002cf2969073f7f201b3b2',
     'wiki.ilo.vec': 'c0e43835a3f4e0033ea5d7c6ff189982b2f26a05',
     'wiki.id.vec': 'c49d5c9bec89114599427f6c12a5bda2e5523dfd',
     'wiki.ia.vec': '2a348dc924638efc20c34785852b0837364aed76',
     'wiki.ie.vec': '01b0d11c0e7397418e73853d220e97bdcf7a8961',
     'wiki.iu.vec': 'ed77a1d7b0faeeb1352b1c4fc1e69971e1e21174',
     'wiki.ik.vec': '4d5d4f7a6426720e07d0faeb51b5babfa4acf44a',
     'wiki.ga.vec': 'caaa5b2167a499893313ac1aa38416a6a0fe9a24',
     'wiki.it.vec': 'ac4a985e85ffae48047034e2603d804bf126caa9',
     'wiki.jam.vec': '6d51e384c56330097c2531fdbf4e74418909e388',
     'wiki.ja.vec': '7a2b1af1e46d795410692a002e40fa3085135f69',
     'wiki.jv.vec': '2ff7927d3ff04b8208133497b3778ede00ea463f',
     'wiki.kbd.vec': 'f5b8dbe47a7fae702232b5680b070ef6e865539e',
     'wiki.kab.vec': 'e3b73d41267d8d4cd42f6cc5a0c05dc4e021bf74',
     'wiki.xal.vec': 'b738222d84cb8c8fdb2b30a7219aa5d3bdc2f61c',
     'wiki.kn.vec': '32763f4f860f0d081f3aabf3e7d17b7858e7d877',
     'wiki.kr.vec': 'c919463e96e4fe36dd5bd73be0c5cd144d4d4f91',
     'wiki.pam.vec': '8fbd31e70d0ca0c61eb1a152efaa8ecb29180967',
     'wiki.krc.vec': '0c6ef043d51e5f337a309804f1db180fa0bb2cb8',
     'wiki.kaa.vec': 'd990d3b9bd511d2d630f923099a6b9110231b2ed',
     'wiki.ks.vec': 'f0a69830a3f661c107503772cc6bd5e345f0c8d6',
     'wiki.csb.vec': '649cb2692f08414987c875dc331022567d367497',
     'wiki.kk.vec': '6343b2b31bad2e13d03a110b91c38fab4adc01cd',
     'wiki.km.vec': '64f7fff1df90b1f7241b232e901f76223a3719e0',
     'wiki.ki.vec': 'c4e373e2ea13f7fa1e95b0733365e4b3fc8b2cc8',
     'wiki.rw.vec': 'af2ec410da6519a86ba21004c8b4c7fde768a91c',
     'wiki.ky.vec': '13b0ae3f23822317a0243bd9182105c631c834b3',
     'wiki.rn.vec': '9df628e8c25d928d3e9d127b624f79fd99ff8f4e',
     'wiki.kv.vec': '164dc44d701b9d606a45f0b0446076adc3858dca',
     'wiki.koi.vec': '4001f0617fe0fdd3b22116b304f497b7b16c6e4c',
     'wiki.kg.vec': '379575f4c6e1c4b73e311ddf01b7a85afd047d7c',
     'wiki.ko.vec': '042c85a788c2778cca538cf716b8a78f0d7fa823',
     'wiki.kj.vec': 'adf29c1a3aa5dd53d85e04d68aa11a26c0eaf6c8',
     'wiki.ku.vec': '4d3a2401527dd9ba6be2b0cd31f6cd3edebadce9',
     'wiki.ckb.vec': 'adb2fef309f1d93f429442b9c16c1564192c58f3',
     'wiki.lad.vec': 'c510e520cde97050bf1cbeb36f2b90e6348ceed4',
     'wiki.lbe.vec': 'e72e5ea054334580f72fbe446a726d2b4962851d',
     'wiki.lo.vec': '7c83f82b80c49b8eab21f62ecdb3681b8bda40a6',
     'wiki.ltg.vec': 'ec2f13d1290bd54afcaa74569e66e43e9bfef264',
     'wiki.la.vec': '9ea6286a0581084533db8d6ee96e0b7d15166543',
     'wiki.lv.vec': 'ef6b549f96e22718f513d47a611d3d6bc001a164',
     'wiki.lez.vec': '8e579b984a500ad89fc66767bfd7319766bd669b',
     'wiki.lij.vec': '4ff5bb405c820e4119f0636efc301da15a08c00a',
     'wiki.li.vec': '0fb9ec4ac93676d8ef651692062bc3d7f6ae0843',
     'wiki.ln.vec': '70b6a286b42958e25cb80824e0d8f1aee2de6dde',
     'wiki.lt.vec': '58d3ebef24e5e31be1a8318b45c08ebb16ad775a',
     'wiki.olo.vec': 'cbadb4cada4dc579d0becdac93dfb479d76bf6c8',
     'wiki.jbo.vec': 'c90481946aa4b6b304528292612ae620f6549f3e',
     'wiki.lmo.vec': 'a89414d9ceee4823622258f18936f67faf7e06e7',
     'wiki.nds.vec': '7bf293149c08226e05bcf0442ac6e601162b9ffd',
     'wiki.dsb.vec': 'e49a647a441fbf011ac5411dd6005e8725b9a65d',
     'wiki.lg.vec': 'b096f5248dfbb343dc4696c97ea253510e1c4ef9',
     'wiki.lb.vec': 'b146f23628c84e64314a35a5b6cc65a33777e22d',
     'wiki.mk.vec': '85a3d3f13fa88ffde023d2326c65bdded4983dff',
     'wiki.mai.vec': '7f513ff36e485b19f91f83b30c32dd82e9e497f6',
     'wiki.mg.vec': '0808252740909d6129f672584311263e7b2adadc',
     'wiki.ms.vec': '458e1a079799a54cdc0a7b78c7fa1729d2683a6d',
     'wiki.ml.vec': '2b70fe76e8cf199a18551de782784a21e8db0b66',
     'wiki.mt.vec': '81f4c1d84dd4cc4276d59cb903fcc9aba46be981',
     'wiki.gv.vec': '993a7ee31bdacc91763dad656aa6c2947b873473',
     'wiki.mi.vec': 'e8acf9c7c2ab840a192c563aa776201a88e4ca89',
     'wiki.mr.vec': '2cd6cf88bfdfb24850d345749ce0cfea8d65829e',
     'wiki.mh.vec': '8c5dbbcb8ad08b9c8b39151fa56d553d116d1b5a',
     'wiki.mzn.vec': 'aefad49237808acab99e1ca8eeaaf531666f261d',
     'wiki.mhr.vec': '39f62e292336cabc364f0d1913540b881b406393',
     'wiki.cdo.vec': '95e8196bf76323dbabab1b8a49ba4d677af3ccea',
     'wiki.zh_min_nan.vec': 'f91ccb013e200bb7ed560082ddf4bdd9c2f315bb',
     'wiki.min.vec': '3bb0fa596cf27a1d165c55684bebdc8d40cb8ad7',
     'wiki.xmf.vec': 'dc1923cfd1a7002d5d60426b60e6756854ab4a14',
     'wiki.mwl.vec': '3d10a218242b94fcc3981aa3beb012b701827a55',
     'wiki.mdf.vec': 'b16099ce0283a241339716eac41cfd99fdea7f36',
     'wiki.mo.vec': '9824ebe366bc52d84e66d1c0cc72b5f7ebb46110',
     'wiki.mn.vec': '7cef7ecdf9d98484d9b598b25d0e717dba6acfd9',
     'wiki.mus.vec': 'bb94534fdeee4df77ae3e27c252c8874f69a307d',
     'wiki.nah.vec': 'c52e01cf4479fb7ec91ef39f298e8f97aeb6496e',
     'wiki.na.vec': 'fbe1444b21e1a5885a619cf2a8607fcefca3c8db',
     'wiki.nv.vec': 'f5a6ea213bfe95c82cb22b53b4965df8b67ffeab',
     'wiki.ng.vec': '8577634e236133980243be0a6fb3c02ad2bb5290',
     'wiki.nap.vec': '6c9bd8ce1e85ee679b25189fd6f6d36afb119b6c',
     'wiki.ne.vec': '1045d7876f947cd4602d9ca79f7c4323a5d3a52d',
     'wiki.new.vec': '51f6c0b4ef1aee9fad4ab1cb69a7479db35e39a5',
     'wiki.pih.vec': 'a6a867cef441a06c926205daa9e405aaf58e8f63',
     'wiki.nrm.vec': 'b4cb941b126b26fa045c5fc75a490a31a969101c',
     'wiki.frr.vec': 'cde62af939cb2de35e341cef2c74813802a58ed4',
     'wiki.lrc.vec': 'c1ae4fb79a19d44bfe8f601f0a30fbec841fa612',
     'wiki.se.vec': 'f46b35ee6b893c2f12dd1b929bbc2b8120cbcd8d',
     'wiki.nso.vec': 'a906271509c2b343df35d1471509492bbfa883aa',
     'wiki.no.vec': 'd52e8019d7cc48569c8c3b514d2b1bd10261b5c0',
     'wiki.nn.vec': '35aeab89ffeca0377accbbd3bf18b81913c75448',
     'wiki.nov.vec': '5455c6e8463b1c43dd073e3e177702fb9a1dd834',
     'wiki.ii.vec': '755a6b8ffa664e342c2ab72847af343c47f46c70',
     'wiki.oc.vec': 'cc1833492899d75571148c2c305591f53d63f0b1',
     'wiki.cu.vec': 'e8eb72eb7fbc224b62ed32dbd897c8c7f6cc5c0a',
     'wiki.or.vec': 'a6b120fe536b6c0133b077dca0043c3bc97eef0b',
     'wiki.om.vec': '91789a8d9f9284f7e71e4bb8d9a60eae4af4adca',
     'wiki.os.vec': '791b26cc300e9a1f0a08c7b2213a264e41ce30d6',
     'wiki.pfl.vec': '0ad9b7f3ae13f909f12835107432fee4c4ed3031',
     'wiki.pi.vec': '07a5d05e5363e8b8b132220a71de4bdc0a623cfc',
     'wiki.pag.vec': '03f71faf060c4eb33802275279967349c0337553',
     'wiki.pap.vec': '8cd98267cc55a4f9de80212e29651ddf7a9e83fd',
     'wiki.ps.vec': '64f1bec5d5b937289199ceae2e1da6557ce48852',
     'wiki.pdc.vec': '401e24d0fb9b0ae9e06a5c700684361f58727fcf',
     'wiki.fa.vec': '09b6cc685c895c66b853af9617787d3ab0891e2c',
     'wiki.pcd.vec': 'd2e8e7321b6f1bce94c563cb8ef8af2b45cc3e48',
     'wiki.pms.vec': 'e30bda8d33d61db43243c157b9ac2feeaff316c8',
     'wiki.pl.vec': 'd031adb6f83eda0364a861dcbf5ef779b5951c0b',
     'wiki.pnt.vec': 'a9efbf962a895e1d08dde5fd56797dd03abb421e',
     'wiki.pt.vec': '7f11ebdb0cbf5929b38319f1e977d2c13bcd741b',
     'wiki.qu.vec': '58de8c8290e8bc8f2a6a677312e28457113437b2',
     'wiki.ksh.vec': '4c3bb4f12073532b6fb7cc6c2be5e53319ef5b65',
     'wiki.rmy.vec': '309fb92222b03f3bd4f2260c02bbd1e3f3d3aba7',
     'wiki.ro.vec': 'c088ea2752d5ec8b42e32410c191a14839ae8a1f',
     'wiki.rm.vec': '5d3144b47a0dd98648a6df0636384ab2a010ad7b',
     'wiki.ru.vec': '7514a2c60ee4118abb451ed32a0d61cb52dec384',
     'wiki.rue.vec': 'fe539e0ea0bbbfd3ee06bd0c5521a035c7361ec5',
     'wiki.sah.vec': '202470467194a1cbdcd571b14ef68371a29b38d9',
     'wiki.sm.vec': '88c2c57ca483626b052403418cb4372d72352bc9',
     'wiki.bat_smg.vec': 'cb3aef58da2011183b39fca64cabf3d9d7a62f4b',
     'wiki.sg.vec': '7b9c8294c060bd10839650afd1f247b950aa819d',
     'wiki.sa.vec': '7fed78d1d7674453b9876ee99aeeeba85ea46699',
     'wiki.sc.vec': 'dba8dc7754ef04b1ba0cd702d94eea9575cde91c',
     'wiki.stq.vec': '1bf88af29f1d86cac16042a5bea6b1651c96a8c1',
     'wiki.sco.vec': '4625a5ad90a57f994be9b3aa4f8f3ecda941a821',
     'wiki.gd.vec': 'f4b513598a1bf0f0d5b6521ea8ce363e9596cb97',
     'wiki.sr.vec': '3cf09f476f55a92fdd2880f7ba336656ab232736',
     'wiki.sh.vec': '016691ecb26ace442731d92b1265e5c6c3d8ca5f',
     'wiki.st.vec': '963646055d12873b1c83b0eef8649ecaf473d42e',
     'wiki.sn.vec': '8dbb1019dcc8f842a8c0f550295ae697f8e1b7e0',
     'wiki.scn.vec': 'bde043a235551e1643506774c5d9b61ecf2fc424',
     'wiki.szl.vec': '0573cf888ec70b459b0596d34814fe60fd69f190',
     'wiki.simple.vec': '55267c50fbdf4e4ae0fbbda5c73830a379d68795',
     'wiki.sd.vec': '36852d1253496e598fbd9b9009f07f454a6bea5b',
     'wiki.si.vec': 'd05ed6a0bc1ee56e5d2e5f881d47372095f6eb0c',
     'wiki.sk.vec': '98759aacf7352d49a51390fae02030776510ae13',
     'wiki.sl.vec': 'b26997c0ed1de26a47b11efdc26ac1e7f189fa54',
     'wiki.so.vec': '294756b60b03fe57cb08abd8d677d6a717b40bc8',
     'wiki.azb.vec': 'e23af0a436b97434813c3cb14ed114cc5b352faa',
     'wiki.es.vec': '2f41401aa0925167176bcd7a6770423d891dfef5',
     'wiki.srn.vec': 'faee05e550f5b08809a9ae5586ac4b08c9a1c359',
     'wiki.su.vec': '25e864495acb6d280bab0e62480f68550c9ceed4',
     'wiki.sw.vec': '8e70d207dbbd14e60a48e260a23fbf284a8e9f06',
     'wiki.ss.vec': '488546a3b2f88f549c50ae9f32f1997cc441b039',
     'wiki.sv.vec': 'eab83ae36701139696477b91b6e8d292ef175053',
     'wiki.tl.vec': 'd508e229ced7201510999e76d583de3ff2339d8b',
     'wiki.ty.vec': 'b881f60b8c75a71864d9847a17961d368f3058fc',
     'wiki.tg.vec': '6a5cd5bfe571ca0359b66d21bf6950553213f42d',
     'wiki.ta.vec': 'b66b5358527b1f3a6a421ab26464a3c1e75e18af',
     'wiki.roa_tara.vec': 'b3fcb01ff0bac53a0ba08c5c0c411f26ee83a95a',
     'wiki.tt.vec': '913bb3a11da6f8142b3bbec3ef065162d9350f1d',
     'wiki.te.vec': 'e71dcf3cc45da1bcdae5e431324025bd2026d0c8',
     'wiki.tet.vec': 'f38fe0e76b9b08ff652689eeee42c4fdadd9a47e',
     'wiki.th.vec': '1d6e0d525392a1042d017534f6c320c5a0afd345',
     'wiki.bo.vec': '2e9358e03dcfa09da23d2e1499d84b10348fd8a9',
     'wiki.ti.vec': 'c769fbc99bbb4138a40231e573685c7948d4a4c4',
     'wiki.tpi.vec': '407b96d235f54f3e0be9dc23a3bab89c6593a621',
     'wiki.to.vec': '64d512665b55e9ef9a3915e8167347be79310fa0',
     'wiki.ts.vec': '00f8229e2f230afd388221c0f823a1de9fc0e443',
     'wiki.tn.vec': '39f45f3fa86645bb25c54150204abcd51cc1048c',
     'wiki.tcy.vec': '388b1d89642fcc790b688e9643b3d19e14d66f40',
     'wiki.tum.vec': 'bfbe43364724af882a520d2edcc2ce049c7357cd',
     'wiki.tr.vec': '13234aa1bf5f99e81d933482b3b83c3e4bf6c85e',
     'wiki.tk.vec': '33ae577f77d339ab7a0dff88855b8d5c974d0aef',
     'wiki.tyv.vec': 'e8f9a36dc58e4108c553f96e247a877a099ab5ba',
     'wiki.tw.vec': 'f329b667d70d9f0b753e55e1b1579b5a5191d3bd',
     'wiki.udm.vec': '336a8526f22e177faac69573661dc9c3ce36591f',
     'wiki.uk.vec': '77f7737b9f88eac2b3e130ea8abb8886336fd0c6',
     'wiki.hsb.vec': '3dc7830544c58535bed308c552d609e13b973502',
     'wiki.ur.vec': 'cb8132102152a958df72bd3e25f1a72abb4c9c76',
     'wiki.ug.vec': '586d2febafaf17c9187c599ffd7b96e559103c34',
     'wiki.uz.vec': '11c3a76dae12b454f693811e33ae2e60015743e2',
     'wiki.ve.vec': 'b7d2947501de1c30a9f8496d5efae20c051104e1',
     'wiki.vec.vec': 'ae4b055fba21974e56beecab3a95f9dc24a62fd0',
     'wiki.vep.vec': 'a38a781fde24f4d7b52aa8bc450b9949dd4e1808',
     'wiki.vi.vec': 'bc84245b52b2e212e28dc6856c0693ce9845a9c5',
     'wiki.vo.vec': 'c830988b6965bfce2f932b1be193f7d1f755f411',
     'wiki.fiu_vro.vec': '168a71a2b1c478e6810fa5dce9612d8bf8a273dc',
     'wiki.wa.vec': '18f9ca1a585e1d18c3630029141a2e19d7d34a8e',
     'wiki.war.vec': '1f5d443d6f612b59a53820dd6f39fd886a6ad30f',
     'wiki.cy.vec': '32d976a9bfc4dd6e39328c906eead0f597bd9e25',
     'wiki.vls.vec': '07e8636908c057b9870ce4b98c7130d460cf882a',
     'wiki.fy.vec': 'd4beef537b7ff142a3986513879ff51a9ec14a7b',
     'wiki.pnb.vec': '35f38862d3d83012d6db7baa8a4105e3e0a416e7',
     'wiki.wo.vec': '2ad96a7a9e640bc0dbcf316b1f414b92802dcb8e',
     'wiki.wuu.vec': 'e1cbae1d3ad52329d0f36ada764016fbacf07049',
     'wiki.xh.vec': 'bf37f741b0b75953281d11df2b4d80100df9e666',
     'wiki.yi.vec': '299d61958b7dcc38774768f1489121384726d860',
     'wiki.yo.vec': 'e35c8aff2924ba07936be9d0d94bd298f09702a4',
     'wiki.diq.vec': '77f3c370d1d77806fafe368cf788af550ff607dd',
     'wiki.zea.vec': 'ee12db26aab3f2b3b2745a298ef414e7aeb5a058',
     'wiki.za.vec': 'e3a0e58bd2e5b1891c71f1f7e37ff71997a20361',
     'wiki.zu.vec': '4b244b9697a8280e6646842c5fc81bb3a6bc8ec7',
     'wiki-news-300d-1M.vec': '11cac9efe6f599e659be182f5766d6fbd5b1cab9',
     'wiki-news-300d-1M-subword.vec': '717a3058e0ba5ef3cde52c3df0d4f0f60b0a113a',
     'crawl-300d-2M.vec': '9b556504d099a6c01f3dd76b88775d02cb2f1946',
     'wiki.multi.ar.vec': 'f1f12cc9d629382af574a3db74fe49c2fd615c8f',
     'wiki.multi.bg.vec': '22470e664e4b35761a33c64433ea2f0c12140673',
     'wiki.multi.ca.vec': 'bc8d98b4d86d740d1985d73d211d887d561bcdd7',
     'wiki.multi.cs.vec': '17358b62e63f96b0479d6a70e9235a0421493884',
     'wiki.multi.da.vec': 'ebc75f428714d26fb1fa31accce49ad3b31e273b',
     'wiki.multi.de.vec': 'b9a63406aedf4446b467b94d12674bfe4723b52d',
     'wiki.multi.el.vec': '03d33db85bf83f35b943ce93b18c02fa98a0bc05',
     'wiki.multi.en.vec': '696719afdbe470ee4a2eb668229486dba1df19cc',
     'wiki.multi.es.vec': '98c9e35564ec57fee5dbc6155890150452f45d3f',
     'wiki.multi.et.vec': 'db10189093387e853f2fd3978770e1cc7bc07820',
     'wiki.multi.fi.vec': '746916885a1c7d4ec3f139a32cf267f9e15f5363',
     'wiki.multi.fr.vec': 'fe1535827b631d934beb02f8d36ba901b2c94a46',
     'wiki.multi.he.vec': '6dd112f018165317da22971a2b6fdb2a15dafa91',
     'wiki.multi.hr.vec': 'ff9f23cf595ec8dd93cd93c6b48049730c34253b',
     'wiki.multi.hu.vec': '6da405c9b048f3cbb990bfb29ef149f0430aa2e7',
     'wiki.multi.id.vec': '34edadab182682198c37ade8538530c545635742',
     'wiki.multi.it.vec': 'c55802bd73d46a6fc86771097670e02a70b5d46d',
     'wiki.multi.mk.vec': 'cec8550503ebca0bdc7ad11f2c15085b7072a990',
     'wiki.multi.nl.vec': 'c3f45a5fe8a8bc213cdf35dce51651b752ca60c4',
     'wiki.multi.no.vec': '105236df530c8fc2ce5b1e2550a2059bbc46fc28',
     'wiki.multi.pl.vec': '676eb5acb22982c0c9a7d6e4c90d26730c6d120e',
     'wiki.multi.pt.vec': '625b0a5384873c79a5dcfff5ee3fde49a3a65013',
     'wiki.multi.ro.vec': '82bd59674509b69f988f9870e3a291836ba43e84',
     'wiki.multi.ru.vec': 'a7d9c5f2ab2abb448a5111d352caa921adabe830',
     'wiki.multi.sk.vec': '98d849ee77f0320472cc5afa002bfde129be7089',
     'wiki.multi.sl.vec': 'fb5cfb8a9c44380d74fb21ddd204e820c4e05c31',
     'wiki.multi.sv.vec': '95d6cc3ba23dffff9be6adb467b617dd57780cb2',
     'wiki.multi.tr.vec': 'ecb0e353eaccba3fcacc6994d93065934ef429e9',
     'wiki.multi.uk.vec': '35f4f5a1ead8bd66bcaf865021fc3aae94456ab6',
     'wiki.multi.vi.vec': 'b1abe06360e1d65a0db65dd41ead7b2f9d651ea0'}
