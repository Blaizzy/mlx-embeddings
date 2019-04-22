# frozen_string_literal: true

require "requirements/java_requirement"
require "fileutils"

describe JavaRequirement do
  subject { described_class.new(%w[1.8]) }

  let(:java_home) { mktmpdir }

  before do
    FileUtils.mkdir java_home/"bin"
    FileUtils.touch java_home/"bin/java"
    allow(subject).to receive(:preferred_java).and_return(java_home/"bin/java")
  end

  specify "Apple Java environment" do
    expect(subject).to be_satisfied

    expect(ENV).to receive(:prepend_path)
    expect(ENV).to receive(:append_to_cflags)

    subject.modify_build_environment
    expect(ENV["JAVA_HOME"]).to eq(java_home.to_s)
  end

  specify "Oracle Java environment" do
    expect(subject).to be_satisfied

    FileUtils.mkdir java_home/"include"
    expect(ENV).to receive(:prepend_path)
    expect(ENV).to receive(:append_to_cflags).twice

    subject.modify_build_environment
    expect(ENV["JAVA_HOME"]).to eq(java_home.to_s)
  end
end
