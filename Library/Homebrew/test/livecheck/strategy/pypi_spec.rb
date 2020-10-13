# typed: false
# frozen_string_literal: true

require "livecheck/strategy/pypi"

describe Homebrew::Livecheck::Strategy::Pypi do
  subject(:pypi) { described_class }

  let(:pypi_url) { "https://files.pythonhosted.org/packages/ab/cd/efg/hij-1.2.3.tar.gz" }
  let(:non_pypi_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a PyPI URL" do
      expect(pypi.match?(pypi_url)).to be true
    end

    it "returns false if the argument provided is not a PyPI URL" do
      expect(pypi.match?(non_pypi_url)).to be false
    end
  end
end
