# typed: false
# frozen_string_literal: true

require "livecheck/strategy/npm"

describe Homebrew::Livecheck::Strategy::Npm do
  subject(:npm) { described_class }

  let(:npm_url) { "https://registry.npmjs.org/abc/-/def-1.2.3.tgz" }
  let(:npm_scoped_url) { "https://registry.npmjs.org/@example/abc/-/def-1.2.3.tgz" }
  let(:non_npm_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is an npm URL" do
      expect(npm.match?(npm_url)).to be true
      expect(npm.match?(npm_scoped_url)).to be true
    end

    it "returns false if the argument provided is not an npm URL" do
      expect(npm.match?(non_npm_url)).to be false
    end
  end
end
