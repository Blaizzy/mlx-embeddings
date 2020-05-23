# frozen_string_literal: true

require "extend/ARGV"

describe HomebrewArgvExtension do
  subject { argv.extend(described_class) }

  let(:argv) { ["mxcl"] }

  describe "#value" do
    let(:argv) { ["--foo=", "--bar=ab"] }

    it "returns the value for a given string" do
      expect(subject.value("foo")).to eq ""
      expect(subject.value("bar")).to eq "ab"
    end

    it "returns nil if there is no matching argument" do
      expect(subject.value("baz")).to be nil
    end
  end
end
