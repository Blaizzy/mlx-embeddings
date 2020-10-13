# typed: true
# frozen_string_literal: true

require "irb"

# @private
module IRB
  def self.parse_opts(argv: nil); end

  def self.start_within(binding)
    unless @setup_done
      setup(nil, argv: [])
      @setup_done = true
    end

    workspace = WorkSpace.new(binding)
    irb = Irb.new(workspace)

    @CONF[:IRB_RC]&.call(irb.context)
    @CONF[:MAIN_CONTEXT] = irb.context

    trap("SIGINT") do
      irb.signal_handle
    end

    begin
      catch(:IRB_EXIT) do
        irb.eval_input
      end
    ensure
      irb_at_exit
    end
  end
end
