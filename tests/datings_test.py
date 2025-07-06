import pytest

from lxml import etree


@pytest.fixture
def sample_xml():
    xml_str = """
<macrogenesis xmlns="http://www.faustedition.net/ns"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://www.faustedition.net/ns https://faustedition.uni-wuerzburg.de/xml/schema/macrogenesis.xsd">
   <!--  / 25/XVIII,7,1 /  / V H1a (Findbuch), V H39 (Landeck) /  -->
   <date notBefore="1831-04-25">
      <comment>Bezug auf Tagebuch auf Hs. Konzept zu Tagebuchnotiz zum 24.4.1831. gehörte
         ursprünglich mit 25/XVIII,8,2 und 25/XVIII,8,3 zu einem Blatt.</comment>
      <source uri="faust://bibliography/landeck1981">S.85-86</source>
      <item uri="faust://document/landeck/2_V_H.39"/>
   </date>
   <!-- geprüft -->
   <!--  / 25/XVIII,7,1a /  / (oS) /  -->
   <!--  / 25/XVIII,7,2 /  / V H1 (V H.1I Landeck) /  -->
   <date notAfter="1831-05-04">
      <comment>Bezug auf Tgb. Tgb vom 04.05.1831 (Abschluss des 5. Akts). .</comment>
      <source uri="faust://bibliography/landeck1981">S.100-101</source>
      <item uri="faust://document/landeck/2_V_H.1I"/>
   </date>
   <!-- geprüft -->
   <date notBefore="1831-04-30" notAfter="1831-05-17">
      <comment>April-Mai 1831</comment>
      <source uri="faust://bibliography/hertz1932">S.262</source>
      <item uri="faust://document/wa/2_V_H.1"/>
   </date>
   <!-- geprüft -->
   <relation name="temp-pre">
      <source uri="faust://bibliography/hertz1932">S.262</source>
      <item uri="faust://document/wa/2_V_H.2"/>
      <item uri="faust://document/landeck/2_V_H.1I"/>
   </relation>
</macrogenesis>
    """
    return etree.fromstring(xml_str)

@pytest.mark.skip()
def test_base_graph():
    G = datings.base_graph()
    assert G is not None
