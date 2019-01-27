require "hardware"

describe Hardware::CPU do
  describe "::type" do
    let(:cpu_types) {
      [
        :intel,
        :dunno,
      ]
    }

    it "returns the current CPU's type as a symbol, or :dunno if it cannot be detected" do
      expect(cpu_types).to include(described_class.type)
    end
  end

  describe "::family" do
    let(:cpu_families) {
      [
        :core,
        :core2,
        :penryn,
        :nehalem,
        :arrandale,
        :sandybridge,
        :ivybridge,
        :haswell,
        :broadwell,
        :skylake,
        :kabylake,
        :dunno,
      ]
    }

    it "returns the current CPU's family name as a symbol, or :dunno if it cannot be detected" do
      expect(cpu_families).to include described_class.family
    end
  end
end
