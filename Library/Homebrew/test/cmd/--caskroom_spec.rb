# typed: false
# frozen_string_literal: true

describe "brew --caskroom", :integration_test do
  let(:local_transmission) {
    Cask::CaskLoader.load(cask_path("local-transmission"))
  }

  let(:local_caffeine) {
    Cask::CaskLoader.load(cask_path("local-caffeine"))
  }

  it "outputs Homebrew's caskroom" do
    expect { brew "--caskroom" }
      .to output("#{HOMEBREW_PREFIX/"Caskroom"}\n").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end

  it "outputs the caskroom path of casks" do
    expect { brew "--caskroom", cask_path("local-transmission"), cask_path("local-caffeine") }
      .to output("#{HOMEBREW_PREFIX/"Caskroom"/"local-transmission"}\n" \
                 "#{HOMEBREW_PREFIX/"Caskroom"/"local-caffeine\n"}").to_stdout
      .and not_to_output.to_stderr
      .and be_a_success
  end
end
