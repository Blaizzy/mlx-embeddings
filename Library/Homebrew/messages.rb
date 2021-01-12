# typed: true
# frozen_string_literal: true

# A {Messages} object collects messages that may need to be displayed together
# at the end of a multi-step `brew` command run.
class Messages
  extend T::Sig

  attr_reader :caveats, :formula_count, :install_times

  sig { void }
  def initialize
    @caveats = []
    @formula_count = 0
    @install_times = []
  end

  def record_caveats(f, caveats)
    @caveats.push(formula: f.name, caveats: caveats)
  end

  def formula_installed(f, elapsed_time)
    @formula_count += 1
    @install_times.push(formula: f.name, time: elapsed_time)
  end

  def display_messages(force_caveats: false, display_times: false)
    display_caveats(force: force_caveats)
    display_install_times if display_times
  end

  def display_caveats(force: false)
    return if @formula_count.zero?
    return if @formula_count == 1 && !force
    return if @caveats.empty?

    oh1 "Caveats"
    @caveats.each do |c|
      ohai c[:formula], c[:caveats]
    end
  end

  def display_install_times
    return if install_times.empty?

    oh1 "Installation times"
    install_times.each do |t|
      puts format("%<formula>-20s %<time>10.3f s", t)
    end
  end
end
