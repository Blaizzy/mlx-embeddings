# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"

describe "brew fetch" do
  let(:local_transmission) do
    Cask::CaskLoader.load(cask_path("local-transmission"))
  end

  it_behaves_like "parseable arguments"

  it "downloads the Formula's URL", :integration_test do
    setup_test_formula "testball"

    expect { brew "fetch", "testball" }.to be_a_success

    expect(HOMEBREW_CACHE/"testball--0.1.tbz").to be_a_symlink
    expect(HOMEBREW_CACHE/"testball--0.1.tbz").to exist
  end

  it "prevents double fetch (without nuking existing installation)", :integration_test do
    setup_test_formula "testball"
    brew "fetch", "testball"

    old_ctime = File.stat(HOMEBREW_CACHE/"testball--0.1.tbz").ctime

    expect { brew "fetch", "testball", "--no-quarantine" }.to be_a_success
    new_ctime = File.stat(HOMEBREW_CACHE/"testball--0.1.tbz").ctime

    expect(old_ctime.to_i).to eq(new_ctime.to_i)
  end

  it "allows double fetch with --force", :integration_test do
    setup_test_formula "testball"
    brew "fetch", "testball"

    old_ctime = File.stat(HOMEBREW_CACHE/"testball--0.1.tbz").ctime
    sleep(1)

    expect { brew "fetch", "testball", "--force", "--no-quarantine" }.to be_a_success
    new_ctime = File.stat(HOMEBREW_CACHE/"testball--0.1.tbz").ctime

    expect(new_ctime.to_i).to be > old_ctime.to_i
  end
end
