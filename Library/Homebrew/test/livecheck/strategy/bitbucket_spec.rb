# typed: false
# frozen_string_literal: true

require "livecheck/strategy/bitbucket"

describe Homebrew::Livecheck::Strategy::Bitbucket do
  subject(:bitbucket) { described_class }

  let(:bitbucket_url) { "https://bitbucket.org/abc/def/get/1.2.3.tar.gz" }
  let(:non_bitbucket_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true for a Bitbucket URL" do
      expect(bitbucket.match?(bitbucket_url)).to be true
    end

    it "returns false for a non-Bitbucket URL" do
      expect(bitbucket.match?(non_bitbucket_url)).to be false
    end
  end
end
