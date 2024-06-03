# frozen_string_literal: true

require "requirements/macos_requirement"

RSpec.describe MacOSRequirement do
  subject(:requirement) { described_class.new }

  let(:big_sur_major) { MacOSVersion.new("11.0") }

  describe "#satisfied?" do
    it "returns true on macOS" do
      expect(requirement.satisfied?).to eq OS.mac?
    end

    it "supports version symbols", :needs_macos do
      requirement = described_class.new([MacOS.version.to_sym])
      expect(requirement).to be_satisfied
    end

    it "supports maximum versions", :needs_macos do
      requirement = described_class.new([:catalina], comparator: "<=")
      expect(requirement.satisfied?).to eq MacOS.version <= :catalina
    end
  end

  specify "#allows?" do
    max_requirement = described_class.new([:mojave], comparator: "<=")
    min_requirement = described_class.new([:catalina], comparator: ">=")
    exact_requirement = described_class.new([:big_sur], comparator: "==")
    range_requirement = described_class.new([[:monterey, :big_sur]], comparator: "==")
    expect(max_requirement.allows?(big_sur_major)).to be false
    expect(min_requirement.allows?(big_sur_major)).to be true
    expect(exact_requirement.allows?(big_sur_major)).to be true
    expect(range_requirement.allows?(big_sur_major)).to be true
  end
end
