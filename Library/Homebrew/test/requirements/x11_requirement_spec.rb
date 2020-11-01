# typed: false
# frozen_string_literal: true

require "cli/args"
require "requirements/x11_requirement"

describe X11Requirement do
  subject(:requirement) { described_class.new([]) }

  let(:default_name) { "x11" }

  describe "#name" do
    it "defaults to x11" do
      expect(requirement.name).to eq(default_name)
    end
  end

  describe "#eql?" do
    it "returns true if the requirements are equal" do
      other = described_class.new
      expect(requirement).to eql(other)
    end
  end

  describe "#modify_build_environment" do
    it "calls ENV#x11" do
      allow(requirement).to receive(:satisfied?).and_return(true)
      expect(ENV).to receive(:x11)
      requirement.modify_build_environment
    end
  end

  describe "#satisfied?", :needs_macos do
    it "returns true if X11 is installed" do
      expect(MacOS::XQuartz).to receive(:version).and_return("2.7.5")
      expect(MacOS::XQuartz).to receive(:installed?).and_return(true)
      expect(requirement).to be_satisfied
    end

    it "returns false if X11 is not installed" do
      expect(MacOS::XQuartz).to receive(:installed?).and_return(false)
      expect(requirement).not_to be_satisfied
    end
  end
end
