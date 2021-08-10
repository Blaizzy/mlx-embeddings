# typed: false
# frozen_string_literal: true

require "livecheck/strategy/apache"

describe Homebrew::Livecheck::Strategy::Apache do
  subject(:apache) { described_class }

  let(:apache_url) { "https://www.apache.org/dyn/closer.lua?path=abc/1.2.3/def-1.2.3.tar.gz" }
  let(:non_apache_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true for an Apache URL" do
      expect(apache.match?(apache_url)).to be true
    end

    it "returns false for a non-Apache URL" do
      expect(apache.match?(non_apache_url)).to be false
    end
  end
end
