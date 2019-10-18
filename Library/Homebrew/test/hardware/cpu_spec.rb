# frozen_string_literal: true

require "hardware"

describe Hardware::CPU do
  describe "::type" do
    let(:cpu_types) {
      [
        :arm,
        :intel,
        :ppc,
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
        :arm,
        :arrandale,
        :atom,
        :broadwell,
        :core,
        :core2,
        :dothan,
        :haswell,
        :ivybridge,
        :kabylake,
        :merom,
        :nehalem,
        :penryn,
        :prescott,
        :presler,
        :sandybridge,
        :skylake,
        :westmere,
        :dunno,
      ]
    }

    it "returns the current CPU's family name as a symbol, or :dunno if it cannot be detected" do
      expect(cpu_families).to include described_class.family
    end
  end
end
