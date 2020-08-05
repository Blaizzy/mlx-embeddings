# frozen_string_literal: true

require "formula"
require "livecheck"

describe Livecheck do
  let(:f) do
    formula do
      url "https://brew.sh/test-0.1.tbz"
    end
  end
  let(:livecheckable) { described_class.new(f) }

  describe "#regex" do
    it "returns nil if unset" do
      expect(livecheckable.regex).to be nil
    end

    it "returns the Regex if set" do
      livecheckable.regex(/foo/)
      expect(livecheckable.regex).to eq(/foo/)
    end
  end

  describe "#skip" do
    it "sets the instance variable skip to true and skip_msg to nil when the argument is not present" do
      livecheckable.skip
      expect(livecheckable.instance_variable_get(:@skip)).to be true
      expect(livecheckable.instance_variable_get(:@skip_msg)).to be nil
    end

    it "sets the instance variable skip to true and skip_msg to the argument when present" do
      livecheckable.skip("foo")
      expect(livecheckable.instance_variable_get(:@skip)).to be true
      expect(livecheckable.instance_variable_get(:@skip_msg)).to eq("foo")
    end
  end

  describe "#skip?" do
    it "returns the value of the instance variable skip" do
      expect(livecheckable.skip?).to be false
      livecheckable.skip
      expect(livecheckable.skip?).to be true
    end
  end

  describe "#strategy" do
    it "returns nil if not set" do
      expect(livecheckable.strategy).to be nil
    end

    it "returns the Symbol if set" do
      livecheckable.strategy(:page_match)
      expect(livecheckable.strategy).to eq(:page_match)
    end

    it "raises a TypeError if the argument isn't a Symbol" do
      expect {
        livecheckable.strategy("page_match")
      }.to raise_error(TypeError, "Livecheck#strategy expects a Symbol")
    end
  end

  describe "#url" do
    it "returns nil if unset" do
      expect(livecheckable.url).to be nil
    end

    it "returns the URL if set" do
      livecheckable.url("foo")
      expect(livecheckable.url).to eq("foo")
    end
  end

  describe "#to_hash" do
    it "returns a Hash of all instance variables" do
      expect(livecheckable.to_hash).to eq(
        {
          "regex"    => nil,
          "skip"     => false,
          "skip_msg" => nil,
          "strategy" => nil,
          "url"      => nil,
        },
      )
    end
  end
end
