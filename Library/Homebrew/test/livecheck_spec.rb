# typed: false
# frozen_string_literal: true

require "formula"
require "livecheck"

describe Livecheck do
  HOMEPAGE_URL = "https://example.com/"
  STABLE_URL = "https://example.com/example-1.2.3.tar.gz"
  HEAD_URL = "https://example.com/example.git"
  CASK_URL = "https://example.com/example-1.2.3.dmg"

  let(:f) do
    formula do
      homepage HOMEPAGE_URL
      url STABLE_URL
      head HEAD_URL
    end
  end
  let(:livecheckable_f) { described_class.new(f) }

  let(:c) do
    Cask::CaskLoader.load(+<<-RUBY)
      cask "test" do
        version "0.0.1,2"

        url CASK_URL
        name "Test"
        desc "Test cask"
        homepage HOMEPAGE_URL
      end
    RUBY
  end
  let(:livecheckable_c) { described_class.new(c) }

  describe "#regex" do
    it "returns nil if not set" do
      expect(livecheckable_f.regex).to be nil
    end

    it "returns the Regexp if set" do
      livecheckable_f.regex(/foo/)
      expect(livecheckable_f.regex).to eq(/foo/)
    end

    it "raises a TypeError if the argument isn't a Regexp" do
      expect {
        livecheckable_f.regex("foo")
      }.to raise_error(TypeError, "Livecheck#regex expects a Regexp")
    end
  end

  describe "#skip" do
    it "sets @skip to true when no argument is provided" do
      expect(livecheckable_f.skip).to be true
      expect(livecheckable_f.instance_variable_get(:@skip)).to be true
      expect(livecheckable_f.instance_variable_get(:@skip_msg)).to be nil
    end

    it "sets @skip to true and @skip_msg to the provided String" do
      expect(livecheckable_f.skip("foo")).to be true
      expect(livecheckable_f.instance_variable_get(:@skip)).to be true
      expect(livecheckable_f.instance_variable_get(:@skip_msg)).to eq("foo")
    end

    it "raises a TypeError if the argument isn't a String" do
      expect {
        livecheckable_f.skip(/foo/)
      }.to raise_error(TypeError, "Livecheck#skip expects a String")
    end
  end

  describe "#skip?" do
    it "returns the value of @skip" do
      expect(livecheckable_f.skip?).to be false

      livecheckable_f.skip
      expect(livecheckable_f.skip?).to be true
    end
  end

  describe "#strategy" do
    it "returns nil if not set" do
      expect(livecheckable_f.strategy).to be nil
    end

    it "returns the Symbol if set" do
      livecheckable_f.strategy(:page_match)
      expect(livecheckable_f.strategy).to eq(:page_match)
    end

    it "raises a TypeError if the argument isn't a Symbol" do
      expect {
        livecheckable_f.strategy("page_match")
      }.to raise_error(TypeError, "Livecheck#strategy expects a Symbol")
    end
  end

  describe "#url" do
    it "returns nil if not set" do
      expect(livecheckable_f.url).to be nil
    end

    it "returns a string when set to a string" do
      livecheckable_f.url("https://brew.sh")
      expect(livecheckable_f.url).to eq("https://brew.sh")
    end

    it "returns the URL symbol if valid" do
      livecheckable_f.url(:head)
      expect(livecheckable_f.url).to eq(:head)

      livecheckable_f.url(:homepage)
      expect(livecheckable_f.url).to eq(:homepage)

      livecheckable_f.url(:stable)
      expect(livecheckable_f.url).to eq(:stable)

      livecheckable_c.url(:url)
      expect(livecheckable_c.url).to eq(:url)
    end

    it "raises a TypeError if the argument isn't a String or valid Symbol" do
      expect {
        livecheckable_f.url(/foo/)
      }.to raise_error(TypeError, "Livecheck#url expects a String or valid Symbol")
    end
  end

  describe "#to_hash" do
    it "returns a Hash of all instance variables" do
      expect(livecheckable_f.to_hash).to eq(
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
