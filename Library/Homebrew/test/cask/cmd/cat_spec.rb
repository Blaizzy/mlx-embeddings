# typed: false
# frozen_string_literal: true

require_relative "shared_examples/requires_cask_token"
require_relative "shared_examples/invalid_option"

describe Cask::Cmd::Cat, :cask do
  it_behaves_like "a command that requires a Cask token"
  it_behaves_like "a command that handles invalid options"

  describe "given a basic Cask" do
    let(:basic_cask_content) {
      <<~'RUBY'
        cask "basic-cask" do
          version "1.2.3"
          sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"

          url "https://brew.sh/TestCask-#{version}.dmg"
          name "Basic Cask"
          desc "Cask for testing basic functionality"
          homepage "https://brew.sh/"

          app "TestCask.app"
        end
      RUBY
    }
    let(:caffeine_content) {
      <<~'RUBY'
        cask "local-caffeine" do
          version "1.2.3"
          sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"

          url "file://#{TEST_FIXTURE_DIR}/cask/caffeine.zip"
          homepage "https://brew.sh/"

          app "Caffeine.app"
        end
      RUBY
    }

    it "displays the Cask file content about the specified Cask" do
      expect {
        described_class.run("basic-cask")
      }.to output(basic_cask_content).to_stdout
    end

    it "can display multiple Casks" do
      expect {
        described_class.run("basic-cask", "local-caffeine")
      }.to output(basic_cask_content + caffeine_content).to_stdout
    end
  end

  it "raises an exception when the Cask does not exist" do
    expect { described_class.run("notacask") }
      .to raise_error(Cask::CaskUnavailableError, /is unavailable/)
  end
end
