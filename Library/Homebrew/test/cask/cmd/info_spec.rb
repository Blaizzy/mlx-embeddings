# typed: false
# frozen_string_literal: true

require_relative "shared_examples/requires_cask_token"
require_relative "shared_examples/invalid_option"
require "utils"

describe Cask::Cmd::Info, :cask do
  it_behaves_like "a command that requires a Cask token"
  it_behaves_like "a command that handles invalid options"

  it "displays some nice info about the specified Cask" do
    expect {
      described_class.run("local-transmission")
    }.to output(<<~EOS).to_stdout
      local-transmission: 2.61
      https://transmissionbt.com/
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/local-transmission.rb
      ==> Name
      Transmission
      ==> Description
      BitTorrent client
      ==> Artifacts
      Transmission.app (App)
    EOS
  end

  it "prints auto_updates if the Cask has `auto_updates true`" do
    expect {
      described_class.run("with-auto-updates")
    }.to output(<<~EOS).to_stdout
      with-auto-updates: 1.0 (auto_updates)
      https://brew.sh/autoupdates
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/with-auto-updates.rb
      ==> Name
      AutoUpdates
      ==> Description
      None
      ==> Artifacts
      AutoUpdates.app (App)
    EOS
  end

  describe "given multiple Casks" do
    let(:expected_output) {
      <<~EOS
        local-caffeine: 1.2.3
        https://brew.sh/
        Not installed
        From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/local-caffeine.rb
        ==> Name
        None
        ==> Description
        None
        ==> Artifacts
        Caffeine.app (App)

        local-transmission: 2.61
        https://transmissionbt.com/
        Not installed
        From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/local-transmission.rb
        ==> Name
        Transmission
        ==> Description
        BitTorrent client
        ==> Artifacts
        Transmission.app (App)
      EOS
    }

    it "displays the info" do
      expect {
        described_class.run("local-caffeine", "local-transmission")
      }.to output(expected_output).to_stdout
    end
  end

  it "prints caveats if the Cask provided one" do
    expect {
      described_class.run("with-caveats")
    }.to output(<<~EOS).to_stdout
      with-caveats: 1.2.3
      https://brew.sh/
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/with-caveats.rb
      ==> Name
      None
      ==> Description
      None
      ==> Artifacts
      Caffeine.app (App)
      ==> Caveats
      Here are some things you might want to know.

      Cask token: with-caveats

      Custom text via puts followed by DSL-generated text:
      To use with-caveats, you may need to add the /custom/path/bin directory
      to your PATH environment variable, e.g. (for bash shell):
        export PATH=/custom/path/bin:"$PATH"

    EOS
  end

  it 'does not print "Caveats" section divider if the caveats block has no output' do
    expect {
      described_class.run("with-conditional-caveats")
    }.to output(<<~EOS).to_stdout
      with-conditional-caveats: 1.2.3
      https://brew.sh/
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/with-conditional-caveats.rb
      ==> Name
      None
      ==> Description
      None
      ==> Artifacts
      Caffeine.app (App)
    EOS
  end

  it "prints languages specified in the Cask" do
    expect {
      described_class.run("with-languages")
    }.to output(<<~EOS).to_stdout
      with-languages: 1.2.3
      https://brew.sh/
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/with-languages.rb
      ==> Name
      None
      ==> Description
      None
      ==> Languages
      zh, en-US
      ==> Artifacts
      Caffeine.app (App)
    EOS
  end

  it 'does not print "Languages" section divider if the languages block has no output' do
    expect {
      described_class.run("without-languages")
    }.to output(<<~EOS).to_stdout
      without-languages: 1.2.3
      https://brew.sh/
      Not installed
      From: https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/without-languages.rb
      ==> Name
      None
      ==> Description
      None
      ==> Artifacts
      Caffeine.app (App)
    EOS
  end

  it "can run be run with a url twice and returns analytics", :needs_network do
    skip "Receiving a 416 when fetching docker.rb"
    analytics = {
      "analytics" => {
        "install" => {
          "30d" => { "docker" => 1000 }, "90d" => { "docker" => 2000 }, "365d" => { "docker" => 3000 }
        },
      },
    }
    expect(Utils::Analytics).to receive(:formulae_brew_sh_json).twice.with("cask/docker.json")
    .and_return(analytics)
    expect {
      described_class.run("https://raw.githubusercontent.com/Homebrew/homebrew-cask" \
                          "/d0b2c58652ae5eff20a7a4ac93292a08b250912b/Casks/docker.rb")
      described_class.run("https://raw.githubusercontent.com/Homebrew/homebrew-cask" \
                          "/d0b2c58652ae5eff20a7a4ac93292a08b250912b/Casks/docker.rb")
    }.to output(<<~EOS).to_stdout
      ==> Downloading https://raw.githubusercontent.com/Homebrew/homebrew-cask/d0b2c58652ae5eff20a7a4ac93292a08b250912b/Casks/docker.rb.
      docker: 2.0.0.2-ce-mac81,30215 (auto_updates)
      https://www.docker.com/community-edition
      Not installed
      ==> Names
      Docker Community Edition
      Docker CE
      ==> Description
      None
      ==> Artifacts
      Docker.app (App)
      ==> Analytics
      install: 1,000 (30 days), 2,000 (90 days), 3,000 (365 days)
      ==> Downloading https://raw.githubusercontent.com/Homebrew/homebrew-cask/d0b2c58652ae5eff20a7a4ac93292a08b250912b/Casks/docker.rb.
      docker: 2.0.0.2-ce-mac81,30215 (auto_updates)
      https://www.docker.com/community-edition
      Not installed
      ==> Names
      Docker Community Edition
      Docker CE
      ==> Description
      None
      ==> Artifacts
      Docker.app (App)
      ==> Analytics
      install: 1,000 (30 days), 2,000 (90 days), 3,000 (365 days)
    EOS
  end
end
