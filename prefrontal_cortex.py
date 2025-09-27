import numpy as np
from brain_region_base import BrainRegion

class PrefrontalCortex(BrainRegion):
    def process(self, input_signal, neurotransmitters, internal_state=None):
        """
        Simulates processing in the prefrontal cortex based on neurotransmitter modulation.

        Parameters:
        - input_signal: list or array of input values representing sensory or cognitive stimuli.
        - neurotransmitters: dictionary containing neurotransmitter levels, e.g., {'dopamine': 1.0, 'glutamate': 1.0}
        - internal_state: optional parameter for future expansion (e.g., working memory state, stress level)

        Returns:
        - normalized output signal after neurotransmitter modulation.

        Scientific Notes:
        -----------------
        - Dopamine plays a critical role in prefrontal cortex function, particularly in cognitive tasks such as
          working memory, planning, decision-making, and response inhibition.
        - The effect of dopamine follows an inverse-U shaped relationship: both insufficient and excessive dopamine
          levels impair cognitive performance. Optimal dopamine levels enhance signal-to-noise ratio and task efficiency.
        - D1 dopamine receptors are especially important for working memory and are densely expressed in the lateral
          prefrontal cortex. Activation of D1 receptors improves neural responsiveness to relevant stimuli.
        - Glutamate is the primary excitatory neurotransmitter and contributes to cortical activation and synaptic plasticity.
        - Stress and aging can alter dopamine levels in the prefrontal cortex, leading to cognitive decline.
        - Genetic polymorphisms (e.g., COMT Val/Met variants) affect dopamine metabolism and influence individual
          differences in cognitive performance and drug response.
        - Traditional practices such as zazen and meditation may help regulate dopamine levels in the prefrontal cortex,
          potentially supporting self-control and cognitive stability. This is supported by primate studies showing that
          lateral prefrontal neurons are actively involved in self-control tasks during trained behavioral paradigms.
        - Classification rule switching and response inhibition are associated with specific subregions of the prefrontal cortex:
          the left lateral posterior ventral area is critical for switching classification rules, while the frontopolar cortex
          (prefrontal pole) is important for suppressing responses based on outdated rules. Primate studies show that neurons
          in these regions encode the current rule and evaluate feedback (correct vs incorrect), supporting flexible behavior.
        - The orbitofrontal cortex (BA11) is involved in value-based decision-making and reward evaluation. It modulates
          the relative weighting of expected outcomes and adjusts behavioral strategies accordingly. Dysfunction in BA11
          may lead to fixed reward sensitivity, impairing adaptive learning and reverse updating of value representations.

        Future Extensions:
        ------------------
        - Include dynamic dopamine regulation based on stress or task difficulty.
        - Model D1 receptor-specific modulation and inverse-U shaped performance curves.
        - Simulate genetic variability (e.g., COMT polymorphisms) and pharmacological interventions.
        - Implement rule switching and response inhibition mechanisms based on lateral and frontopolar prefrontal subregions.
        - Add orbitofrontal (BA11) module to simulate value ratio adjustment and feedback-based learning flexibility.
        """

        # Retrieve neurotransmitter levels with default values
        dopamine = neurotransmitters.get('dopamine', 1.0)
        glutamate = neurotransmitters.get('glutamate', 1.0)

        # Modulate input signal based on neurotransmitter levels
        # Dopamine and glutamate jointly influence signal strength and cortical responsiveness
        weighted_input = np.array(input_signal) * dopamine * glutamate

        # Apply exponential transformation to simulate nonlinear neural activation
        exp_input = np.exp(weighted_input)

        # Normalize to simulate competitive encoding and probabilistic output
        return exp_input / np.sum(exp_input)