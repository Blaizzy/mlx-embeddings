# frozen_string_literal: true

require "version"
require "os/mac/version"

describe OS::Mac::Version do
  subject { described_class.new("10.14") }

  specify "comparison with Symbol" do
    expect(subject).to be > :high_sierra
    expect(subject).to be == :mojave
    expect(subject).to be === :mojave # rubocop:disable Style/CaseEquality
    expect(subject).to be < :catalina
  end

  specify "comparison with Fixnum" do
    expect(subject).to be > 10
    expect(subject).to be < 11
  end

  specify "comparison with Float" do
    expect(subject).to be > 10.13
    expect(subject).to be < 10.15
  end

  specify "comparison with String" do
    expect(subject).to be > "10.3"
    expect(subject).to be == "10.14"
    expect(subject).to be === "10.14" # rubocop:disable Style/CaseEquality
    expect(subject).to be < "10.15"
  end

  specify "comparison with Version" do
    expect(subject).to be > Version.create("10.3")
    expect(subject).to be == Version.create("10.14")
    expect(subject).to be === Version.create("10.14") # rubocop:disable Style/CaseEquality
    expect(subject).to be < Version.create("10.15")
  end

  specify "#from_symbol" do
    expect(described_class.from_symbol(:mojave)).to eq(subject)
    expect { described_class.from_symbol(:foo) }
      .to raise_error(MacOSVersionError, "unknown or unsupported macOS version: :foo")
  end

  specify "#pretty_name" do
    expect(described_class.new("10.11").pretty_name).to eq("El Capitan")
    expect(described_class.new("10.14").pretty_name).to eq("Mojave")
    expect(described_class.new("10.10").pretty_name).to eq("Yosemite")
  end

  specify "#requires_nehalem_cpu?" do
    expect(Hardware::CPU).to receive(:type).at_least(:twice).and_return(:intel)
    expect(described_class.new("10.14").requires_nehalem_cpu?).to be true
    expect(described_class.new("10.12").requires_nehalem_cpu?).to be false
  end
end
