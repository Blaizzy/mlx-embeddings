# frozen_string_literal: true

require "version"
require "os/mac/version"

describe OS::Mac::Version do
  subject { described_class.new("10.10") }

  specify "comparison with Symbol" do
    expect(subject).to be > :mavericks
    expect(subject).to be == :yosemite
    expect(subject).to be === :yosemite # rubocop:disable Style/CaseEquality
    expect(subject).to be < :el_capitan
  end

  specify "comparison with Fixnum" do
    expect(subject).to be > 10
    expect(subject).to be < 11
  end

  specify "comparison with Float" do
    expect(subject).to be > 10.9
    expect(subject).to be < 10.11
  end

  specify "comparison with String" do
    expect(subject).to be > "10.9"
    expect(subject).to be == "10.10"
    expect(subject).to be === "10.10" # rubocop:disable Style/CaseEquality
    expect(subject).to be < "10.11"
  end

  specify "comparison with Version" do
    expect(subject).to be > Version.create("10.9")
    expect(subject).to be == Version.create("10.10")
    expect(subject).to be === Version.create("10.10") # rubocop:disable Style/CaseEquality
    expect(subject).to be < Version.create("10.11")
  end

  specify "#from_symbol" do
    expect(described_class.from_symbol(:yosemite)).to eq(subject)
    expect { described_class.from_symbol(:foo) }
      .to raise_error(ArgumentError)
  end

  specify "#pretty_name" do
    expect(described_class.new("10.11").pretty_name).to eq("El Capitan")
    expect(described_class.new("10.14").pretty_name).to eq("Mojave")
    expect(described_class.new("10.10").pretty_name).to eq("Yosemite")
  end

  specify "#requires_nehalem_cpu?" do
    expect(described_class.new("10.14").requires_nehalem_cpu?).to be true
    expect(described_class.new("10.12").requires_nehalem_cpu?).to be false
  end
end
